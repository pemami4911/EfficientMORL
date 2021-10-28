#!/bin/bash

NUM_GPUS=1
SEED=1201
DDP_PORT=29530
ENV=clevrtex # clevr6, multi_dsprites, tetrominoes
MODEL=EMORL # EMORL, SlotAttention
DATA_PATH=/blue/ranka/pemami/clevrtex #YOUR_DATA_PATH
BATCH_SIZE=8 #32 / NUM_GPUS
OUT_DIR=/blue/ranka/pemami/experiments #YOUR_RESULTS_DIR
HOST_NODE_ADDR='127.0.0.1:'$DDP_PORT

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

export SACRED_OBSERVATORY=""

cd ..
pwd 

# Optionally, indicate available GPU IDs with CUDA_VISIBLE_DEVICES
python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS --rdzv_endpoint=$HOST_NODE_ADDR train.py with configs/train/$ENV/$MODEL.json texdataset.path=$DATA_PATH seed=$SEED training.batch_size=$BATCH_SIZE training.run_suffix=emorl-$ENV-seed-$SEED training.DDP_port=$DDP_PORT training.out_dir=$OUT_DIR training.tqdm=True