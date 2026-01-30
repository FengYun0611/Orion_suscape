#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

# Get the absolute path of the script directory
# This works correctly even when called through sbatch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="$SCRIPT_DIR/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    "$SCRIPT_DIR/test.py" $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
