import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from sacred import Experiment, cli_option
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from lib.datasets import ds
from lib.datasets import HdF5Dataset
from lib.model import net
from lib.model import EfficientMORL, SlotAttention
from lib.geco import GECO
from lib.visualization import visualize_output, visualize_slots


@cli_option('-r','--local_rank')
def local_rank_option(args, run):
    run.info['local_rank'] = args

ex = Experiment('TRAINING', ingredients=[ds, net], additional_cli_options=[local_rank_option])

@ex.config
def cfg():
    training = {
            'DDP_port': 29500,  # torch.distributed config
            'batch_size': 16,  # training mini-batch size
            'num_workers': 8,  # pytorch dataloader workers
            'mode': 'train',  # dataset
            'model': 'EfficientMORL',  # model name
            'iters': 500000,  # gradient steps to take
            'refinement_curriculum': [(-1,3), (100000,1), (200000,1)],   # (step,I): Update refinement iters I at step 
            'lr': 3e-4,  # Adam LR
            'warmup': 10000,  # LR warmup
            'decay_rate': 0.5,  # LR decay
            'decay_steps': 100000,  # LR decay steps
            'kl_beta_init': 1,  # kl_beta from beta-VAE
            'use_scheduler': False,  # LR scheduler
            'tensorboard_freq': 100,  # how often to write to TB
            'checkpoint_freq': 25000,  # save checkpoints every % steps
            'load_from_checkpoint': False,  # whether to load from a checkpoint or not
            'checkpoint': '',  # name of .pth file to load model state
            'run_suffix': 'debug',  # string to append to run name
            'out_dir': 'experiments',  # where output folders for run results go
            'use_geco': True,  # Use GECO (Rezende & Viola 2018)
            'clip_grad_norm': True,  # Grad norm clipping to 5.0
            'geco_reconstruction_target': -23000,  # GECO C
            'geco_ema_alpha': 0.99,  # GECO EMA step parameter
            'geco_beta_stepsize': 1e-6,  # GECO Lagrange parameter beta
            'tqdm': False  # Show training progress in CLI
        }


def save_checkpoint(step, kl_beta, model, model_opt, filepath):
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),
        'kl_beta': kl_beta
    }
    torch.save(state, filepath)


@ex.automain
def run(training, seed, _run):

    run_dir = Path(training['out_dir'], 'runs')
    checkpoint_dir = Path(training['out_dir'], 'weights')
    tb_dir = Path(training['out_dir'], 'tb')
    
    # Avoid issues with torch distributed and just create directory structure 
    # beforehand
    # training['out_dir']/runs
    # training['out_dir']/weights
    # training['out_dir']/tb    
    for dir_ in [run_dir, checkpoint_dir, tb_dir]:
        if not dir_.exists():
            print(f'Create {dir_} before running!')
            exit(1)

    tb_dbg = tb_dir / training['run_suffix']

    local_rank = 'cuda:{}'.format(_run.info['local_rank'])
    if local_rank == 'cuda:0':
        print(f'Creating SummaryWriter! ({local_rank})')
        writer = SummaryWriter(tb_dbg)
    
    # Fix random seed
    print(f'setting random seed to {seed}')
    # Auto-set by sacred
    # torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # Auto-set by sacred 
    #np.random.seed(seed)
        
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(training['DDP_port'])
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    if training['model'] == 'EfficientMORL':
        model = EfficientMORL(batch_size=training['batch_size'],
                              use_geco=training['use_geco'])
    elif training['model'] == 'SlotAttention':
        model = SlotAttention(batch_size=training['batch_size'])
    else:
        raise RuntimeError('Model {} unknown'.format(training['model']))
    
    model_geco = None
    if training['use_geco']:
        model_geco = GECO(training['geco_reconstruction_target'],
                          training['geco_ema_alpha'])        
    
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    model.train()

    # Optimization
    model_opt = torch.optim.Adam(model.parameters(), lr=training['lr'])
    if training['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                  model_opt,
                  lr_lambda=lambda epoch: 0.5 ** (epoch / 100000)
        )
        scheduler_warmup = GradualWarmupScheduler(model_opt, multiplier=1,
                                                  total_epoch=training['warmup'],
                                                  after_scheduler=scheduler)
    else:
        scheduler_warmup = None

    if not training['load_from_checkpoint']:    
        step = 0 
        kl_beta = training['kl_beta_init']
        checkpoint_step = 0
    else:
        checkpoint = checkpoint_dir / training['checkpoint']
        map_location = {'cuda:0': local_rank}
        state = torch.load(checkpoint, map_location=map_location)
        model.load_state_dict(state['model'])
        model_opt.load_state_dict(state['model_opt'])
        kl_beta = state['kl_beta']
        step = state['step']
        checkpoint_step = step


    tr_dataset = HdF5Dataset(d_set=training['mode'])
    batch_size = training['batch_size']
    tr_sampler = DistributedSampler(dataset=tr_dataset)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
            batch_size=batch_size, sampler=tr_sampler, 
            num_workers=training['num_workers'],
            worker_init_fn=worker_init_fn,
            drop_last=True)
    
    max_iters = training['iters']
    
    print('Num parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    epoch_idx = 0
    refinement_curriculum_counter = 0

    while step <= max_iters:
        
        # Re-shuffle every epoch
        tr_sampler.set_epoch(epoch_idx)

        if training['tqdm'] and local_rank == 'cuda:0':
            data_iter = tqdm(tr_dataloader)
        else:
            data_iter = tr_dataloader

        for batch in data_iter:
            # Update refinement iterations by curriculum
            for rf in range(len(training['refinement_curriculum'])-1,-1,-1):
                if step >= training['refinement_curriculum'][rf][0]:
                    model.module.refinement_iters = training['refinement_curriculum'][rf][1]
                    break
            
            img_batch = batch['imgs'].to(local_rank)
            model_opt.zero_grad()

            # Forward
            if training['model'] == 'SlotAttention':
                out_dict = model(img_batch)
            else:
                out_dict = model(img_batch, model_geco, step, kl_beta)

            # Backward
            total_loss = out_dict['total_loss']
            total_loss.backward()
            if training['use_scheduler']:
                scheduler_warmup.step(step)
            if training['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            model_opt.step()
            
            if training['use_geco']:
                if step == model.module.geco_warm_start:
                    model.module.geco_C_ema = model_geco.init_ema(model.module.geco_C_ema, nll)
                elif step > model.module.geco_warm_start:
                    model.module.geco_C_ema = model_geco.update_ema(model.module.geco_C_ema, nll)
                    model.module.geco_beta = model_geco.step_beta(model.module.geco_C_ema,
                            model.module.geco_beta, training['geco_beta_stepsize'])

            # logging
            if step % training['tensorboard_freq'] == 0 and local_rank == 'cuda:0':
                if training['model'] == 'SlotAttention':
                    writer.add_scalar('train/MSE', total_loss, step)
                    visualize_slots(writer, (img_batch+1)/2., out_dict, step)
                else:
                    writer.add_scalar('train/total_loss', total_loss, step)
                    writer.add_scalar('train/KL', out_dict['kl'], step)
                    writer.add_scalar('train/KL_beta', kl_beta, step)
                    writer.add_scalar('train/NLL', out_dict['nll'], step)
                    visualize_output(writer, (img_batch+1)/2., out_dict,
                                     model.module.stochastic_layers+model.module.refinement_iters,
                                     step)

                if training['use_geco']:
                    writer.add_scalar('train/geco_beta', model.module.geco_beta, step)
                    writer.add_scalar('train/geco_C_ema', model.module.geco_C_ema, step)

                if 'deltas' in out_dict:
                    for refine_iter in range(out_dict['deltas'].shape[0]):
                        writer.add_scalar(f'train/norm_delta_lamda_{refine_iter}',
                                          out_dict['deltas'][refine_iter], step)


            if (step > 0 and step % training['checkpoint_freq'] == 0 and 
                local_rank == 'cuda:0'):
                # Save the model
                prefix = training['run_suffix']
                save_checkpoint(step, kl_beta, model, model_opt, 
                       checkpoint_dir / f'{prefix}-state-{step}.pth')
                        
            if step >= max_iters:
                step += 1
                break
            step += 1
        epoch_idx += 1

    if local_rank == 'cuda:0':
        writer.close()
