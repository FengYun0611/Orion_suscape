# SUScape Fine-tuning Guide for ORION with ShareGPT QA Integration

This guide explains how to fine-tune ORION on the SUScape dataset with ShareGPT QA integration.

## Overview

The fine-tuning process enables ORION to learn how to effectively utilize QA conversation context (q1-q9) to improve trajectory prediction on the SUScape dataset.

## Prerequisites

1. **Pretrained ORION checkpoint**: `ckpts/orion_stage3.pth`
2. **SUScape scene data**: Located at `/lab/haoq_lab/cse12311753/suscape_scenes/`
3. **ShareGPT QA data**: Located at `/lab/haoq_lab/cse12311753/sharegpt_dataset/`
4. **SUScape pkl file**: `data/suscape_infos/suscape_infos_test.pkl` (in the repository)
5. **GPU resources**: 4x A100 GPUs recommended (can adjust batch size for fewer GPUs)

## Quick Start

### Method 1: Automated Script (Recommended)

```bash
# Make the script executable
chmod +x run_suscape_finetune.sh

# Run fine-tuning
./run_suscape_finetune.sh
```

This script will:
1. Automatically split `suscape_infos_test.pkl` into `suscape_infos_train.pkl` (80%) and `suscape_infos_val.pkl` (20%)
2. Start distributed training on 4 GPUs
3. Save checkpoints every epoch
4. Show evaluation command when done

### Method 2: Manual Steps

#### Step 1: Prepare Data Split

```bash
python tools/split_suscape_data.py \
    --input data/suscape_infos/suscape_infos_test.pkl \
    --output-dir data/suscape_infos \
    --train-ratio 0.8 \
    --seed 42
```

This creates:
- `suscape_infos_train.pkl`: 80% of data (for training)
- `suscape_infos_val.pkl`: 20% of data (for validation)

#### Step 2: Run Training

**Multi-GPU (recommended):**
```bash
./adzoo/orion/orion_dist_train.sh \
    adzoo/orion/configs/suscape_finetune.py \
    4 \
    --work-dir work_dirs/orion_suscape_finetune \
    --load-from ckpts/orion_stage3.pth \
    --seed 42
```

**Single GPU:**
```bash
python adzoo/orion/train.py \
    adzoo/orion/configs/suscape_finetune.py \
    --work-dir work_dirs/orion_suscape_finetune \
    --load-from ckpts/orion_stage3.pth \
    --seed 42
```

#### Step 3: Monitor Training

Training logs and tensorboard events are saved in `work_dirs/orion_suscape_finetune/`.

View tensorboard:
```bash
tensorboard --logdir work_dirs/orion_suscape_finetune/
```

#### Step 4: Evaluate Fine-tuned Model

```bash
python adzoo/orion/test.py \
    adzoo/orion/configs/suscape_eval.py \
    work_dirs/orion_suscape_finetune/latest.pth \
    --eval bbox
```

## Configuration Details

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_gpus` | 4 | Number of GPUs |
| `batch_size` | 2 | Per-GPU batch size |
| `num_epochs` | 10 | Training epochs |
| `lr` | 1e-5 | Learning rate (10x lower than pretraining) |
| `num_iters_per_epoch` | 1000 | Iterations per epoch (adjust based on data size) |

### QA Integration

The configuration enables QA-aware training:
- `mix_qa_training=True`: QA conversations are loaded during training
- `use_critical_qa=True`: QA context is fed to LLM
- QA tasks q1-q9 are used for scene understanding

### Data Flow

1. **Dataset loads**: Images + CSV trajectories + ShareGPT QA
2. **QA is tokenized**: Converted to LLM input tokens
3. **Forward pass**: Vision → QT-Former → LLM (with QA context) → Trajectory Decoder
4. **Loss**: L2 loss between predicted and GT trajectories
5. **Backprop**: Model learns to utilize QA for better predictions

## Expected Results

### Training Time
- **4x A100 GPUs**: ~2-3 hours for 10 epochs (depends on dataset size)
- **1x A100 GPU**: ~8-12 hours for 10 epochs

### Performance
After fine-tuning, you should see:
- **L2 error improvement**: 4.92m → 3-4m (estimated)
- **Better QA utilization**: Model learns to leverage VRU detection, traffic signals, and planning advice from QA
- **Domain adaptation**: Model adapts from nuScenes to SUScape

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size in `suscape_finetune.py`:
```python
batch_size = 1  # Was 2
```

### Different number of GPUs
Update `num_gpus` in `suscape_finetune.py` and the training script:
```python
num_gpus = 2  # Was 4
```

### Adjust dataset size
If you have different number of samples, update:
```python
num_iters_per_epoch = <your_dataset_size> // (num_gpus * batch_size)
```

### Resume from checkpoint
```bash
./adzoo/orion/orion_dist_train.sh \
    adzoo/orion/configs/suscape_finetune.py \
    4 \
    --work-dir work_dirs/orion_suscape_finetune \
    --resume-from work_dirs/orion_suscape_finetune/iter_5000.pth
```

## Advanced: Custom Train/Val Split

Modify split ratio:
```bash
python tools/split_suscape_data.py \
    --input /lab/haoq_lab/cse12311753/suscape_scenes/test.pkl \
    --output-dir /lab/haoq_lab/cse12311753/suscape_scenes \
    --train-ratio 0.9 \  # Use 90% for training
    --seed 42
```

## File Structure

After setup, you should have:
```
/lab/haoq_lab/cse12311753/
├── suscape_scenes/
│   ├── test.pkl          # Original data
│   ├── train.pkl         # Training split (created)
│   ├── val.pkl           # Validation split (created)
│   └── scene-XXXXXX/     # Scene directories
├── sharegpt_dataset/
│   ├── suscape_NQA_q1.json
│   ├── suscape_NQA_q2.json
│   └── ...               # q1-q9 QA files
└── ...

work_dirs/
└── orion_suscape_finetune/
    ├── iter_1000.pth
    ├── iter_2000.pth
    ├── latest.pth
    └── tf_logs/          # Tensorboard logs
```

## Next Steps

After fine-tuning:
1. Evaluate on the full test set
2. Compare L2 errors with/without QA fine-tuning
3. Analyze which QA types (VRU detection, traffic signals, etc.) contribute most
4. Consider longer training or hyperparameter tuning for better results

## Support

For issues or questions about fine-tuning, check:
- Training logs: `work_dirs/orion_suscape_finetune/*.log`
- QA loading debug: `python final_qa_debug.py`
- Dataset validation: `python validate_qa_integration.py`
