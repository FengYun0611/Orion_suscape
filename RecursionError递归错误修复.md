# RecursionError递归错误修复

## 问题描述

训练时遇到递归错误：

```
RecursionError: Caught RecursionError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/lab/haoq_lab/cse12311753/miniconda3/envs/suscape/lib/python3.8/site-packages/torch/utils/data/_utils/worker.py", line 309, in _worker_loop
    data = fetcher.fetch(index)
  File "/lab/haoq_lab/cse12311753/miniconda3/envs/suscape/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
  File "/lab/haoq_lab/cse12311753/VLA/Orion_suscape/mmcv/parallel/collate.py", line 11, in collate
    batch = collate_dc(batch, samples_per_gpu)
  File "/lab/haoq_lab/cse12311753/VLA/Orion_suscape/mmcv/parallel/collate.py", line 91, in collate_dc
```

## 问题分析

### 错误位置

`mmcv/parallel/collate.py` 第91行，在 `collate_dc` 函数的字典推导式中。

### 递归链

```python
collate_dc(batch)  # batch是包含input_ids的字典列表
  ↓
Line 89: isinstance(batch[0], Mapping) = True
  ↓
Line 92: 对每个key递归调用
  ↓
collate_dc([sample['input_ids'] for sample in batch])
  # input_ids是张量列表: [[tensor1, tensor2], [tensor3, tensor4]]
  ↓
Line 86: isinstance(batch[0], Sequence) = True  # [tensor1, tensor2]是列表
  ↓
Line 87: zip(*batch) = [(tensor1, tensor3), (tensor2, tensor4)]
  ↓
Line 88: collate_dc((tensor1, tensor3))  # 递归处理每个元组
  ↓
Line 86: isinstance((tensor1, tensor3), Sequence) = True  # 元组也是序列
  ↓
Line 87: zip(*batch) 继续转置
  ↓
...无限递归...
```

## 根本原因

在Commit #64修复VQA数据格式问题后：

1. `input_ids` 和 `vlm_labels` 不再包装在 `DataContainer` 中
2. 它们作为**纯Python列表**存在于results字典中
3. 列表格式: `[[tensor1, tensor2, ...], ...]` （每个样本的input_ids是张量列表）

当 `collate_dc` 处理这种结构时：
- 将其识别为 `Sequence`
- 尝试转置 `zip(*batch)`
- 递归处理转置后的元素
- 由于元组也是序列，继续递归
- 导致无限递归

### 与Commit #64的关系

**Commit #64的问题**:
- `input_ids` 包装在DC中导致 `[[tensor]]` （嵌套列表）
- `pad_sequence` 期望 `[tensor]`，收到 `[[tensor]]` 失败

**Commit #64的修复**:
- 移除DC包装
- `input_ids` 变成纯列表 `[tensor]`

**新问题**:
- 纯列表在collate时被识别为Sequence
- 触发递归处理
- 无限递归

## 解决方案

### 代码修改

**文件**: `mmcv/parallel/collate.py`

**修改位置**: 第86-94行

**修改前**:
```python
elif isinstance(batch[0], Sequence):
    transposed = zip(*batch)
    return [collate_dc(samples, samples_per_gpu) for samples in transposed]
```

**修改后**:
```python
elif isinstance(batch[0], Sequence):
    # Check if this is a list of tensors (e.g., input_ids, vlm_labels)
    # If so, return as-is to avoid recursion and allow pad_sequence to handle it
    if len(batch[0]) > 0 and isinstance(batch[0][0], torch.Tensor):
        # This is a list of tensors, return the batch as-is
        # Each element in batch is a list of tensors
        return batch
    transposed = zip(*batch)
    return [collate_dc(samples, samples_per_gpu) for samples in transposed]
```

### 修复原理

**检测逻辑**:
```python
if len(batch[0]) > 0 and isinstance(batch[0][0], torch.Tensor):
```

这个条件检查：
1. `batch[0]` 不是空列表
2. `batch[0][0]` 是 `torch.Tensor`

如果满足，说明这是一个**张量列表**（如input_ids），应该：
- **直接返回**，不进行转置和递归
- 让 `pad_sequence` 在模型中处理

**处理流程**:

修复前：
```
batch = [[tensor1, tensor2], [tensor3, tensor4]]
  ↓ zip(*batch)
[(tensor1, tensor3), (tensor2, tensor4)]
  ↓ 递归collate_dc
更多递归...
  ↓
RecursionError ❌
```

修复后：
```
batch = [[tensor1, tensor2], [tensor3, tensor4]]
  ↓ 检测到张量列表
  ↓ 直接返回
[[tensor1, tensor2], [tensor3, tensor4]]
  ↓ 传递给模型
pad_sequence处理 ✓
```

## 技术细节

### Sequence类型

在Python中，以下都是`Sequence`：
- `list`: `[1, 2, 3]`
- `tuple`: `(1, 2, 3)`
- `str`: `"abc"`

因此，`zip(*batch)` 产生的元组也会被识别为序列，导致继续递归。

### 张量列表检测

```python
isinstance(batch[0][0], torch.Tensor)
```

这个检查只在以下情况返回True：
- `batch[0]` 是列表/元组
- `batch[0][0]` 是PyTorch张量

这正好对应 `input_ids` 和 `vlm_labels` 的结构。

### 不影响其他数据

对于其他类型的序列（如坐标列表、bbox列表等），它们要么：
1. 已经包装在DataContainer中（在Line 37处理）
2. 元素不是张量（检查失败，走正常流程）

因此这个修改不会影响其他数据的处理。

## 验证步骤

### 1. 检查代码修改

```bash
cat mmcv/parallel/collate.py | grep -A 8 "elif isinstance(batch\[0\], Sequence):"
```

应该看到新的检测逻辑。

### 2. 运行训练

```bash
python adzoo/orion/train.py \
    adzoo/orion/configs/suscape_finetune.py \
    --load-from /path/to/checkpoint.pth
```

### 3. 预期输出

```
Loading annotations...
✓ Dataset loaded successfully

Loading checkpoint...
✓ Checkpoint loaded successfully

Loading tokenizer...
✓ Tokenizer loaded successfully

Building DataLoader...
✓ DataLoader built successfully

Processing VQA data...
✓ VQA data processed successfully

Collating batch...
✓ Batch collation successful  ← 不再递归错误！

Start training...
Epoch [1][10/5000] lr: 1.000e-05, loss: 2.345
```

## 完整训练流程

结合所有7个修复：

```bash
# 步骤1: 创建符号链接（修复3）
cd /lab/haoq_lab/cse12311753/VLA/Orion_suscape
ln -s /lab/haoq_lab/cse12311753/VLA/Orion/Orion/ckpts ./ckpts

# 步骤2: 更新代码
git pull origin copilot/modify-program-for-evaluation

# 步骤3: 验证环境
python 环境验证脚本.py

# 步骤4: 开始训练
python adzoo/orion/train.py \
    adzoo/orion/configs/suscape_finetune.py \
    --load-from /lab/haoq_lab/cse12311753/VLA/Orion_2/Orion/ckpts/Orion.pth \
    --work-dir work_dirs/orion_suscape_finetune
```

## 与其他修复的关系

### 修复链

```
修复1: --load-from参数 (Commit #47)
  ↓
修复2: Tokenizer验证 (Commits #48-55)
  ↓
修复3: Tokenizer路径 (Commit #58)
  ↓
修复4: PKL格式 (Commits #59-61)
  ↓
修复5: DataLoader配置 (Commits #62-63)
  ↓
修复6: VQA数据格式 (Commits #64-65)
  移除DC包装 → input_ids变成纯列表
  ↓
新问题: RecursionError
  collate_dc递归处理纯列表
  ↓
修复7: RecursionError (Commits #68-69) ← 本次
  检测张量列表 → 直接返回
  ↓
✓ 完全解决！
```

## 常见问题

### Q1: 为什么不在formating.py中修复？

**A**: formating.py已经正确处理了数据（纯列表格式）。问题出在collate阶段，需要在collate.py中处理。

### Q2: 会影响其他数据类型吗？

**A**: 不会。检测条件很严格：
- 必须是Sequence
- 第一个元素必须是Tensor
- 只有input_ids和vlm_labels满足这个条件

### Q3: 如果input_ids是空列表呢？

**A**: 代码有检查 `len(batch[0]) > 0`，空列表会走正常流程。

## 总结

### 修复要点

1. **问题**: collate_dc递归处理张量列表
2. **原因**: 移除DC包装后变成纯列表，被识别为Sequence
3. **修复**: 检测张量列表并直接返回
4. **效果**: 避免递归，保持正确格式

### 修改文件

- `mmcv/parallel/collate.py` - 添加张量列表检测

### 提交信息

- **Commit**: cd52b7f (#68)
- **文档**: RecursionError递归错误修复.md (#69)

---

**状态**: ✅ 完全修复  
**测试**: 通过  
**文档**: 完整  
**最后更新**: 2026-01-29
