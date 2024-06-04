#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

CHECKPOINT_PATH=/users/burgerm/iopsstor/petagraph/logs
# VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=<Specify path to file>/gpt2-merges.txt

# DATA_PATH=/users/burgerm/petagraph/data/samples
DATA_PATH=/users/burgerm/petagraph/resources/unitigs_names.txt

GPT_ARGS="
    --num-layers 2 \
    --hidden-size 256 \
    --num-attention-heads 2 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 256 \
    --global-batch-size 256 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --num-workers 16 \
    --dataloader-type single \
    --tokenizer-type Petagraph \
    --data-path $DATA_PATH \
    --vocab-size 8 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --wandb-project petagraph \
    --wandb-exp-name test \
    --log-progress \
    --log-interval 50 \
    --save-interval 10000 \
    --eval-interval 500 \
    --eval-iters 20
"

torchrun $DISTRIBUTED_ARGS pretrain_petagraph.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
