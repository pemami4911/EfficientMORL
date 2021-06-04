import torch
import torchvision


def visualize_slots(writer, batch_data, model_outs, step):
    """
    Render images for each mask and slot reconstruction,
    as well as mask*slot for Slot Attention
    """
    with torch.no_grad():
        batch_size, C, H, W = batch_data.shape
        imgs = batch_data[0]
        
        mask_iter_grid = torchvision.utils.make_grid(model_outs[f'masks'][0].exp())
        mean_iter_grid = torchvision.utils.make_grid(model_outs[f'x_means'][0])
        recon_grid = torch.sum(model_outs[f'masks'][0].exp() * model_outs[f'x_means'][0], 0)

        writer.add_image(f'RGB', mean_iter_grid, step)
        writer.add_image(f'masks', mask_iter_grid, step)
        writer.add_image(f'reconstruction', recon_grid, step)
        writer.add_image('image', imgs, step)


def visualize_output(writer, batch_data, model_outs, num_stochastic_layers, num_refine_iters, step):
    """
    Render images for each mask and slot reconstruction,
    as well as mask*slot, and attention for batch 0 for EfficientMORL
    """
    with torch.no_grad():
        batch_size, C, H, W = batch_data.shape
        imgs = batch_data[0]
        
        i = (num_stochastic_layers+num_refine_iters)-1
        
        mask_iter_grid = torchvision.utils.make_grid(model_outs[f'masks_{i}'][0].exp())
        mean_iter_grid = torchvision.utils.make_grid(model_outs[f'means_{i}'][0])
        recon_grid = torch.sum(model_outs[f'masks_{i}'][0].exp() * model_outs[f'means_{i}'][0], 0)
        attn = torchvision.utils.make_grid(model_outs[f'attn_{num_stochastic_layers-1}'][0].view(-1,1,H,W))

        writer.add_image(f'RGB_level_{i}', mean_iter_grid, step)
        writer.add_image(f'masks_level_{i}', mask_iter_grid, step)
        writer.add_image(f'reconstruction_level_{i}', recon_grid, step)
        writer.add_image(f'attn_level_{num_stochastic_layers-1}', attn, step)
        writer.add_image('image', imgs, step)

