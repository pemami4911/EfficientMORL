#!/bin/bash

NUM_GPUS=1
SEED=1
DDP_PORT=29500
ENV=tetrominoes # clevr6, multi_dsprites, tetrominoes
JSON_file=EMORL # EMORL, SlotAttention, or X_{activeness, dci, preprocessing, viz}
EVAL_TYPE=ARI_MSE_KL  # {ARI_MSE_KL, sample_viz, disentanglement}
DATA_PATH= #YOUR_DATA_PATH
OUT_DIR= #YOUR_RESULTS_DIR
CHECKPOINT=emorl-tetrominoes-seed-1200-state-200000.pth  #the .pth file 
DISENTANGLE_SLOT=0  # for disentanglement viz of a slot

cd ..

python -m torch.distributed.launch --nproc_per_node=1 eval.py with configs/test/$ENV/$EVAL_TYPE/$JSON_file.json dataset.data_path=$DATA_PATH test.DDP_port=$DDP_PORT seed=$SEED test.checkpoint=$CHECKPOINT test.disentangle_slot=$DISENTANGLE_SLOT test.out_dir=$OUT_DIR