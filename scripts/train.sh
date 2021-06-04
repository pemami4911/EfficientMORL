#!/bin/bash

NUM_GPUS=4
SEED=1200
DDP_PORT=29500
ENV=clevr6 # clevr6, multi_dsprites, tetrominoes
MODEL=EMORL # EMORL, SlotAttention
DATA_PATH= #YOUR_DATA_PATH
BATCH_SIZE=8 #32 / NUM_GPUS
OUT_DIR= #YOUR_RESULTS_DIR

if [[ ! -d "$OUT_DIR/weights" ]]
then
    mkdir -p "$OUT_DIR/weights"
fi

if [[ ! -d "$OUT_DIR/runs" ]]
then
    mkdir -p "$OUT_DIR/runs"
fi

if [[ ! -d "$OUT_DIR/tb" ]]
then
    mkdir -p "$OUT_DIR/tb"
fi

cd ..
pwd 

# Optionally, indicate available GPU IDs with CUDA_VISIBLE_DEVICES
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train.py with configs/train/$ENV/$MODEL.json dataset.data_path=$DATA_PATH seed=$SEED training.batch_size=$BATCH_SIZE training.run_suffix=emorl-$ENV-seed-$SEED training.DDP_port=$DDP_PORT training.out_dir=$OUT_DIR training.tqdm=False