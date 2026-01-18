#!/usr/bin/env bash
WORLD_SIZE=$1
GPUS=$2
CONFIG=$3
CHECKPOINT=$4
RANK=${RANK:=0}
MASTER_ADDR=${MASTER_ADDR:='127.0.0.1'}
MASTER_PORT=${MASTER_PORT:='23456'}

# Get the absolute path of the script directory
# This works correctly even when called through sbatch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="$SCRIPT_DIR/../..":$PYTHONPATH \
python  -m torch.distributed.launch --nnodes=${WORLD_SIZE} \
    --node_rank=$RANK --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    "$SCRIPT_DIR/test.py" $CONFIG $CHECKPOINT --launcher pytorch ${@:5} --eval bbox \
    # 2>&1 | tee "${WORK_DIR}/std_out_rank_${RANK}.log"
    