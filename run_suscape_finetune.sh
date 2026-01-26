#!/bin/bash
# ------------------------------------------------------------------------
# SUScape Fine-tuning Script for ORION with ShareGPT QA Integration
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

# Configuration
CONFIG="adzoo/orion/configs/suscape_finetune.py"
WORK_DIR="work_dirs/orion_suscape_finetune"
PRETRAINED_CKPT="ckpts/orion_stage3.pth"
NUM_GPUS=4

# Step 1: Prepare train/val split (if not already done)
echo "============================================"
echo "Step 1: Preparing train/val split"
echo "============================================"

if [ ! -f "/lab/haoq_lab/cse12311753/suscape_scenes/train.pkl" ]; then
    echo "Splitting test.pkl into train.pkl and val.pkl..."
    python tools/split_suscape_data.py \
        --input /lab/haoq_lab/cse12311753/suscape_scenes/test.pkl \
        --output-dir /lab/haoq_lab/cse12311753/suscape_scenes \
        --train-ratio 0.8 \
        --seed 42
else
    echo "train.pkl already exists, skipping split"
fi

echo ""
echo "============================================"
echo "Step 2: Starting ORION fine-tuning"
echo "============================================"

if [ $NUM_GPUS -gt 1 ]; then
    echo "Running distributed training on $NUM_GPUS GPUs..."
    ./adzoo/orion/orion_dist_train.sh \
        $CONFIG \
        $NUM_GPUS \
        --work-dir $WORK_DIR \
        --load-from $PRETRAINED_CKPT \
        --seed 42
else
    echo "Running single GPU training..."
    python adzoo/orion/train.py \
        $CONFIG \
        --work-dir $WORK_DIR \
        --load-from $PRETRAINED_CKPT \
        --seed 42
fi

echo ""
echo "============================================"
echo "Fine-tuning completed!"
echo "============================================"
echo "Checkpoints saved in: $WORK_DIR"
echo ""
echo "To evaluate the fine-tuned model, run:"
echo "python adzoo/orion/test.py \\"
echo "    adzoo/orion/configs/suscape_eval.py \\"
echo "    $WORK_DIR/latest.pth \\"
echo "    --eval bbox"
