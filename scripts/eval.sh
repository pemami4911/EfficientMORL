#!/bin/bash

NUM_GPUS=1
SEED=2
DDP_PORT=29500
ENV=clevr6-128x128 # clevr6, multi_dsprites, tetrominoes
JSON_file=EMORL # EMORL, SlotAttention, or X_{activeness, dci, preprocessing, viz}
EVAL_TYPE=sample_viz  # {ARI_MSE_KL, sample_viz, disentanglement}
DATA_PATH=/blue/ranka/pemami #YOUR_DATA_PATH
OUT_DIR=/blue/ranka/pemami/experiments #YOUR_RESULTS_DIR
#CHECKPOINT=EMORL-clevrtex-seed-3025-full-state-300000.pth  #the .pth file 
CHECKPOINT=EMORL-clevr6-128x128-seed-3023-big-clevr-state-300000.pth
DISENTANGLE_SLOT=0  # for disentanglement viz of a slot
HOST_NODE_ADDR='127.0.0.1:'$DDP_PORT

cd ..

python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS --rdzv_endpoint=$HOST_NODE_ADDR eval.py with configs/test/$ENV/$EVAL_TYPE/$JSON_file.json dataset.data_path=$DATA_PATH test.DDP_port=$DDP_PORT seed=$SEED test.checkpoint=$CHECKPOINT test.disentangle_slot=$DISENTANGLE_SLOT test.out_dir=$OUT_DIR