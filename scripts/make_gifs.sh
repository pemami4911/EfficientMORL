#!/bin/bash

NUM_GPUS=1
SEED=1000
DDP_PORT=29500

DATA_PATH= #YOUR_DATA_PATH
OUT_DIR= #YOUR_RESULTS_DIR
CHECKPOINT= #the .pth file 
ENV=clevr6-96x96 # clevr6-96x96, multi_dsprites, tetrominoes
JSON_file=EMORL # EMORL, SlotAttention, or if clevr6, X_{activeness, dci, preprocessing, viz}
EVAL_TYPE=make_gifs  # {ARI_MSE_KL, sample_viz, disentanglement_viz}

HOST_NODE_ADDR='127.0.0.1:'$DDP_PORT

cd ..

python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS --rdzv_endpoint=$HOST_NODE_ADDR eval.py with configs/test/$ENV/$EVAL_TYPE/$JSON_file.json dataset.data_path=$DATA_PATH test.DDP_port=$DDP_PORT seed=$SEED test.checkpoint=$CHECKPOINT test.out_dir=$OUT_DIR