#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=/users/burgerm/iopsstor/petagraph/logs
# VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=<Specify path to file>/gpt2-merges.txt

# DATA_PATH=/users/burgerm/petagraph/data/samples
DATA_PATH=/users/burgerm/petagraph/resources/unitigs_names.txt

GPT_ARGS="
    --num-layers 2 \
    --hidden-size 256 \
    --num-attention-heads 2 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
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
    --tokenizer-type Petagraph \
    --data-path $DATA_PATH \
    --vocab-size 8 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun pretrain_petagraph.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
