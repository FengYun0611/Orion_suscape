# Troubleshooting Guide for Training Issues

## Quick Diagnosis

If you encounter an error starting with:
```
Traceback (most recent call last):
  File "adzoo/orion/train.py", line 242, in <module>
```

**This is just the first line of the error!** Please provide the **complete error message** including:
- All intermediate lines showing the call stack
- The final error type and message

## Automated Environment Check

Run the environment verification script:

```bash
python 环境验证脚本.py
```

This will check:
- Python version
- Required packages
- Critical files
- Symlinks
- Data files
- Applied code fixes

## Common Error Patterns

### Pattern 1: Import Errors

**Error:**
```python
ModuleNotFoundError: No module named 'torch'
ImportError: cannot import name 'xxx'
```

**Solution:**
```bash
pip install -r requirements.txt
# or
pip install torch torchvision opencv-python mmcv transformers
```

### Pattern 2: File Not Found

**Error:**
```python
FileNotFoundError: [Errno 2] No such file or directory: './ckpts/pretrain_qformer/'
```

**Solution:**
Create symlink (see detailed guide in `Tokenizer路径配置问题解决方案.md`):
```bash
ln -s /path/to/real/ckpts ./ckpts
```

### Pattern 3: HuggingFace Validation Error

**Error:**
```python
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
```

**Solution:**
See `Tokenizer加载错误最终修复.md` - this is fixed in the latest code.

### Pattern 4: Data Format Errors

**Error:**
```python
AttributeError: 'str' object has no attribute 'get'
TypeError: expected Tensor as element 0 in argument 0, but got list
```

**Solution:**
These are fixed in commits #59-65. Update your code:
```bash
git pull origin copilot/modify-program-for-evaluation
```

## Complete Fix Checklist

Ensure all 6 major fixes are applied:

- [ ] **Fix #1**: `--load-from` parameter support
- [ ] **Fix #2**: Tokenizer HuggingFace validation  
- [ ] **Fix #3**: Tokenizer path mismatch (symlink)
- [ ] **Fix #4**: PKL file format handling
- [ ] **Fix #5**: DataLoader configuration
- [ ] **Fix #6**: VQA data format (DC wrapping)

**Verify with:**
```bash
python 环境验证脚本.py
```

## Step-by-Step Debugging

### Step 1: Get Complete Error
Copy the **entire** traceback from `Traceback` to the last line showing the error type and message.

### Step 2: Check Latest Code
```bash
git status
git pull origin copilot/modify-program-for-evaluation
```

### Step 3: Verify Environment
```bash
python 环境验证脚本.py
```

### Step 4: Check File Paths
```bash
# Check symlink
ls -la ./ckpts/pretrain_qformer/

# Check data
ls -la data/suscape_infos/

# Check checkpoint
ls -lh /path/to/your/checkpoint.pth
```

### Step 5: Run with Verbose Logging
Add to your training command:
```bash
python adzoo/orion/train.py \
    adzoo/orion/configs/suscape_finetune.py \
    --load-from /path/to/checkpoint.pth \
    --work-dir work_dirs/test \
    2>&1 | tee training.log
```

## Documentation Index

### Problem-Specific Guides
- `训练参数修复说明.md` - `--load-from` parameter
- `Tokenizer加载错误最终修复.md` - Tokenizer issues
- `Tokenizer路径配置问题解决方案.md` - Path configuration
- `PKL文件格式错误修复说明.md` - PKL format
- `DataLoader配置错误修复.md` - DataLoader config
- `VQA数据格式错误修复.md` - VQA data format

### General Guides
- `README_微调.md` - Getting started
- `微调完整指南.md` - Complete guide
- `所有训练问题完整解决方案总结.md` - All fixes summary
- `训练故障排除指南.md` - This guide (Chinese)

## Reporting New Issues

If you encounter a **new** issue not covered above, please provide:

1. **Complete error traceback** (all lines)
2. **Command you ran**
3. **Environment info:**
   ```bash
   python --version
   pip list | grep -E 'torch|mmcv|opencv|transformers'
   ```
4. **Config file** you're using
5. **Output from:**
   ```bash
   python 环境验证脚本.py
   ```

## Quick Solutions

### I just want to start training!

```bash
# Step 1: Create symlink if needed
cd /path/to/Orion_suscape
ln -s /path/to/real/ckpts ./ckpts

# Step 2: Verify environment
python 环境验证脚本.py

# Step 3: Run training
python adzoo/orion/train.py \
    adzoo/orion/configs/suscape_finetune.py \
    --load-from /path/to/checkpoint.pth \
    --work-dir work_dirs/orion_suscape_finetune
```

### Training hangs or crashes

1. Check GPU memory: `nvidia-smi`
2. Reduce batch size in config
3. Use fewer GPUs: `--gpus 1`
4. Check logs: `tail -f work_dirs/*/logs/*.log`

### Need to resume training

```bash
python adzoo/orion/train.py \
    adzoo/orion/configs/suscape_finetune.py \
    --resume-from work_dirs/orion_suscape_finetune/latest.pth \
    --work-dir work_dirs/orion_suscape_finetune
```

---

**Remember**: Most issues are due to:
1. Missing files or symlinks
2. Incorrect paths in config
3. Missing dependencies
4. Using old version of code

Check these first!
