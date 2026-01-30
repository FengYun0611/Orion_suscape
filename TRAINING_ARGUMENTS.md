# Training Arguments Guide

## Overview

This guide explains the command-line arguments for `train.py` and how to use them for different training scenarios.

## Key Arguments

### Checkpoint Loading Arguments

#### `--load-from <checkpoint_path>`
**Purpose**: Load pretrained model weights for fine-tuning or transfer learning.

**What it loads**:
- ✓ Model weights only
- ✗ Optimizer state (NOT loaded)
- ✗ Training epoch/iteration (starts from 0)

**Use cases**:
- Fine-tuning on a new dataset
- Transfer learning
- Starting from a pretrained model

**Example**:
```bash
python train.py config.py --load-from /path/to/pretrained.pth
```

#### `--resume-from <checkpoint_path>`
**Purpose**: Resume interrupted training from a checkpoint.

**What it loads**:
- ✓ Model weights
- ✓ Optimizer state
- ✓ Training epoch/iteration
- ✓ Learning rate scheduler state

**Use cases**:
- Continue training after interruption
- Resume from a specific epoch

**Example**:
```bash
python train.py config.py --resume-from /path/to/checkpoint.pth
```

### Other Important Arguments

#### `--work-dir <directory>`
Specify where to save logs and model checkpoints.

```bash
python train.py config.py --work-dir work_dirs/my_experiment
```

#### `--gpus <num_gpus>` or `--gpu-ids <id1 id2 ...>`
Control GPU usage for non-distributed training.

```bash
# Use 2 GPUs
python train.py config.py --gpus 2

# Use specific GPUs
python train.py config.py --gpu-ids 0 2
```

#### `--no-validate`
Skip validation during training (speeds up training).

```bash
python train.py config.py --no-validate
```

#### `--seed <random_seed>`
Set random seed for reproducibility.

```bash
python train.py config.py --seed 42
```

#### `--deterministic`
Enable deterministic mode for CUDNN backend (fully reproducible but slower).

```bash
python train.py config.py --seed 42 --deterministic
```

#### `--autoscale-lr`
Automatically scale learning rate based on number of GPUs.

```bash
python train.py config.py --autoscale-lr
```

## Complete Examples

### Example 1: Fine-tune from Pretrained Weights
```bash
python train.py \
    configs/suscape_finetune.py \
    --load-from ckpts/orion_stage3.pth \
    --work-dir work_dirs/suscape_finetune \
    --seed 42
```

### Example 2: Distributed Training with Pretrained Weights
```bash
./orion_dist_train.sh \
    configs/suscape_finetune.py \
    4 \
    --load-from ckpts/orion_stage3.pth \
    --work-dir work_dirs/suscape_finetune
```

### Example 3: Resume Interrupted Training
```bash
python train.py \
    configs/my_config.py \
    --resume-from work_dirs/my_experiment/epoch_10.pth
```

### Example 4: Training from Scratch
```bash
python train.py \
    configs/my_config.py \
    --work-dir work_dirs/from_scratch \
    --seed 0
```

## Distributed Training

For distributed training, use `orion_dist_train.sh`:

```bash
./orion_dist_train.sh <config> <num_gpus> [additional_args]
```

**Example**:
```bash
./orion_dist_train.sh configs/suscape_finetune.py 4 --load-from ckpts/orion_stage3.pth
```

## Configuration Overrides

### Using `--cfg-options`
Override configuration values from command line:

```bash
python train.py config.py \
    --cfg-options \
    optimizer.lr=0.0001 \
    runner.max_epochs=20
```

**Complex overrides**:
```bash
python train.py config.py \
    --cfg-options \
    data.samples_per_gpu=2 \
    "lr_config.step=[8,11]"
```

## Common Scenarios

### Scenario: Fine-tuning SUScape Dataset

**Step 1**: Prepare data split
```bash
python tools/split_suscape_data.py \
    --input data/suscape_infos/suscape_infos_test.pkl \
    --output-dir data/suscape_infos
```

**Step 2**: Start fine-tuning
```bash
./orion_dist_train.sh \
    adzoo/orion/configs/suscape_finetune.py \
    4 \
    --load-from ckpts/orion_stage3.pth
```

### Scenario: Resume After Crash

```bash
# Find the latest checkpoint
ls work_dirs/my_experiment/*.pth

# Resume from it
python train.py config.py \
    --resume-from work_dirs/my_experiment/latest.pth
```

### Scenario: Quick Test with 1 GPU

```bash
python train.py config.py \
    --gpus 1 \
    --work-dir work_dirs/test_run \
    --no-validate \
    --cfg-options runner.max_epochs=2
```

## Troubleshooting

### Error: "unrecognized arguments: --load-from"
**Solution**: Make sure you're using the updated `train.py` that includes the `--load-from` argument.

### Error: Checkpoint file not found
**Solution**: Verify the path to the checkpoint file:
```bash
ls -lh /path/to/checkpoint.pth
```

### Out of Memory (OOM)
**Solutions**:
- Reduce batch size: `--cfg-options data.samples_per_gpu=1`
- Use fewer GPUs
- Enable gradient checkpointing in config

### Training too slow
**Solutions**:
- Disable validation: `--no-validate`
- Use more GPUs for distributed training
- Enable mixed precision training in config

## Advanced Tips

### Monitoring Training
Use TensorBoard to monitor training:
```bash
tensorboard --logdir work_dirs/my_experiment
```

### Saving Disk Space
Configure checkpoint saving interval in your config file:
```python
checkpoint_config = dict(interval=5)  # Save every 5 epochs
```

### Learning Rate Finding
Use `--autoscale-lr` with different GPU counts to find optimal learning rate:
```bash
# Automatically adjusts LR for 4 GPUs
./orion_dist_train.sh config.py 4 --autoscale-lr
```

## Summary Table

| Argument | Purpose | Example |
|----------|---------|---------|
| `--load-from` | Load pretrained weights | `--load-from ckpts/model.pth` |
| `--resume-from` | Resume training | `--resume-from work_dirs/latest.pth` |
| `--work-dir` | Output directory | `--work-dir work_dirs/exp1` |
| `--gpus` | Number of GPUs | `--gpus 4` |
| `--seed` | Random seed | `--seed 42` |
| `--no-validate` | Skip validation | `--no-validate` |
| `--deterministic` | Reproducible mode | `--deterministic` |
| `--autoscale-lr` | Auto scale LR | `--autoscale-lr` |
| `--cfg-options` | Override config | `--cfg-options lr=0.001` |

---

*Last updated: 2026-01-28*
