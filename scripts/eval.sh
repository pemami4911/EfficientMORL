#!/bin/bash

NUM_GPUS=1
SEED=1
DDP_PORT=29500
ENV='clevr6' # clevr6, multi_dsprites, tetrominoes
JSON_file='EMORL' # EMORL, SlotAttention, or X_{activeness, dci, preprocessing, viz}
EVAL_TYPE='ARI_MSE_KL'  # 
DATA_PATH=#YOUR_DATA_PATH
OUT_DIR=#YOUR_RESULTS_DIR
CHECKPOINT=#the .pth file 
CHECKPOINT_DIR=#location where the .pth file is to be found
DISENTANGLE_SLOT=0

python -m torch.distributed.launch --nproc_per_node=1 eval_models.py with configs/test/$ENV/$EVAL_TYPE/$JSON_file.json dataset.data_path=$DATA_PATH test.DDP_port=$DDP_port seed=$SEED test.checkpoint=$CHECKPOINT test.checkpoint_dir=$CHECKPOINT_DIR test.disentangle_slot=$DISENTANGLE_SLOT
