#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH -C gpu
#SBATCH --time=4:00:00
#SBATCH --job-name=petagraph
#SBATCH --output=/users/burgerm/petagraph/Megatron-LM/slurm_logs/petagraph_%j.out

# Load the modules
module load cray

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16

GPUS_PER_NODE=4
# Change for multinode config
# MASTER_ADDR=localhost
echo "NODELIST="${SLURM_NODELIST}
# will output nid[002066,002068]

first_node=$(echo $SLURM_NODELIST | cut -d',' -f1 | sed 's/nid//')
first_node=$(echo $first_node | sed 's/\[//')

echo "First node: nid$first_node"
master_addr=nid$first_node

export MASTER_ADDR=$master_addr
MASTER_PORT=6000

NNODES=4
# NODE_RANK=0
# # WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# set environment variables needed for pytorch DDP
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # $SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

DISTRIBUTED_ARGS="--nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
	--rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
	--rdzv-backend c10d \
	--max-restarts 0 \
	--role `hostname -s`: \
	-t 3"

CHECKPOINT_PATH=/users/burgerm/iopsstor/petagraph/logs/contigs
TENSORBOARD=/users/burgerm/iopsstor/petagraph/contigs/tensorboard

# VOCAB_FILE=<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=<Specify path to file>/gpt2-merges.txt

# DATA_PATH=/users/burgerm/petagraph/data/samples
# DATA_PATH=/users/burgerm/petagraph/resources/unitigs_names.txt
DATA_PATH=/users/burgerm/petagraph/resources/contigs_names.txt

GPT_ARGS="--num-layers 16 \
    --hidden-size 512 \
    --num-attention-heads 4 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 24 \
    --global-batch-size 384 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32"

DATA_ARGS="--num-workers 16 \
    --dataloader-type single \
    --tokenizer-type Petagraph \
    --data-path $DATA_PATH \
    --vocab-size 8 \
    --split 949,50,1"

OUTPUT_ARGS="--log-validation-ppl-to-tensorboard \
    --log-throughput \
    --tensorboard-dir $TENSORBOARD \
    --wandb-project petagraph \
    --wandb-exp-name test-dist \
    --log-progress \
    --log-interval 50 \
    --save-interval 10000 \
    --eval-interval 500 \
    --eval-iters 20"

command="
cd Megatron-LM && torchrun \
--nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
	--rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
	--rdzv-backend c10d \
	--max-restarts 0 \
	--role `hostname -s`: \
	-t 3 \
    pretrain_petagraph.py $GPT_ARGS $DATA_ARGS $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH"

echo $command


srun -u  --container-writable --environment=petagraph_python_env bash -c "${command}"

