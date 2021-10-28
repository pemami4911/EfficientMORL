#!/bin/bash

NUM_GPUS=8
SEED=1201
DDP_PORT=29530
ENV=clevr6-96x96 # clevr6-96x96, multi_dsprites, tetrominoes
MODEL=EMORL # EMORL, SlotAttention
DATA_PATH= #YOUR_DATA_PATH
BATCH_SIZE=4 #32 / NUM_GPUS
OUT_DIR= #YOUR_RESULTS_DIR
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

# Set this to empty string if sacred runs should not be stored
export SACRED_OBSERVATORY="$OUT_DIR/runs"

cd ..
pwd 

# Optionally, indicate available GPU IDs with CUDA_VISIBLE_DEVICES
python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS --rdzv_endpoint=$HOST_NODE_ADDR train.py with configs/train/$ENV/$MODEL.json dataset.data_path=$DATA_PATH seed=$SEED training.batch_size=$BATCH_SIZE training.run_suffix=emorl-$ENV-seed-$SEED training.DDP_port=$DDP_PORT training.out_dir=$OUT_DIR training.tqdm=False