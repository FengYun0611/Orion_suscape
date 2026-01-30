# SUScape 数据集配置指南 - 针对您的文件结构

根据您提供的文件结构，这里是必要的配置修改说明。

## 您的文件结构

```
suscape_scenes/
├── scene-000000/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── scene-000001/
└── ...

suscape_scene_traj_csv_alldistance_fixyaw/
├── 0.csv      # 对应 scene-000000
├── 1.csv      # 对应 scene-000001
└── ...
```

## 需要修改的配置文件

### 1. 编辑配置文件

打开 `adzoo/orion/configs/suscape_eval.py`，找到以下部分并修改：

```python
# 原来的配置：
dataset_type = "SUScapeOrionDataset"
data_root = "data/suscape_scenes"
info_root = "data/suscape_infos"

# 修改为（根据您的实际路径）：
dataset_type = "SUScapeOrionDataset"
data_root = "data/suscape_scenes"  # 包含 scene-XXXXXX 文件夹的目录
csv_root = "data/suscape_scene_traj_csv_alldistance_fixyaw"  # 包含 CSV 文件的目录
info_root = "data/suscape_infos"
```

**重要**: `csv_root` 参数已经在配置文件中设置好了，默认值为 `"data/suscape_scene_traj_csv_alldistance_fixyaw"`。

### 2. 确认路径正确

根据您的实际文件位置，可能需要调整路径：

- 如果 `suscape_scenes/` 在项目根目录下，使用相对路径：
  ```python
  data_root = "suscape_scenes"
  csv_root = "suscape_scene_traj_csv_alldistance_fixyaw"
  ```

- 如果在 `data/` 目录下，使用：
  ```python
  data_root = "data/suscape_scenes"
  csv_root = "data/suscape_scene_traj_csv_alldistance_fixyaw"
  ```

## 验证数据集

运行验证工具确认配置正确：

```bash
# 对于您的新结构，需要提供两个参数：
python tools/verify_suscape_dataset.py \
    data/suscape_scenes \
    data/suscape_scene_traj_csv_alldistance_fixyaw
```

如果验证成功，会显示：

```
============================================================
SUScape Dataset Structure Verification
============================================================

✓ Found scenes directly in data_root - using NEW structure
  Scenes location: data/suscape_scenes
  CSV location: data/suscape_scene_traj_csv_alldistance_fixyaw
✓ Found X scene directories

  Checking scene: scene-000000
    ✓ CSV format: Valid CSV with ... rows and ... timestamps
    ✓ CAM_FRONT: X images
    ✓ CAM_FRONT_LEFT: X images
    ...

============================================================
SUMMARY
============================================================
Total scenes: X
Valid scenes: X
Invalid scenes: 0

✓ All scenes are valid!
```

## 运行评估

验证通过后，运行评估：

```bash
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

首次运行会自动：
1. 扫描 `data/suscape_scenes/` 目录中的所有 `scene-XXXXXX` 文件夹
2. 从 `data/suscape_scene_traj_csv_alldistance_fixyaw/` 读取对应的 CSV 文件
   - `scene-000000` → `0.csv`
   - `scene-000001` → `1.csv`
   - 依此类推
3. 生成标注文件并保存到 `data/suscape_infos/suscape_infos_test.pkl`
4. 开始评估

## 关键变化说明

### 代码已自动支持您的结构

现在的代码支持两种目录结构：

1. **旧结构**（之前的默认）：
   ```
   data_root/raws/scene-XXXXXX/
       ├── 0.csv
       └── CAM_*/
   ```

2. **新结构**（您的结构）：
   ```
   data_root/scene-XXXXXX/CAM_*/
   csv_root/X.csv
   ```

当设置了 `csv_root` 参数时，代码会自动使用新结构。

### CSV 文件映射规则

- `scene-000000` → `0.csv`
- `scene-000001` → `1.csv`
- `scene-000123` → `123.csv`

场景编号是从场景目录名称中提取的数字（例如从 `scene-000123` 提取 `123`）。

## 常见问题

### Q: 如果我的 CSV 文件名不是数字怎么办？

A: 目前代码假设 CSV 文件名是场景编号（例如 `0.csv`, `1.csv`）。如果您的 CSV 文件名不同，需要修改 `_generate_annotations_new_structure()` 方法中的映射逻辑。

### Q: 场景编号和 CSV 文件名不匹配怎么办？

A: 确保：
- `scene-000000` 对应 `0.csv`
- `scene-000001` 对应 `1.csv`
- 场景目录名中的数字（去掉前导零）应该等于 CSV 文件名中的数字

### Q: 验证工具报错找不到 CSV 文件？

A: 检查：
1. `csv_root` 路径是否正确
2. CSV 文件是否存在且命名正确
3. 场景编号提取是否正确

### Q: 需要重新生成标注吗？

A: 如果您之前用旧结构生成过标注，需要删除旧的标注文件：

```bash
rm data/suscape_infos/suscape_infos_test.pkl
```

然后重新运行评估，系统会使用新结构重新生成标注。

## 总结

对于您的文件结构，只需要：

1. ✅ 确认配置文件中 `csv_root` 已设置（默认已设置）
2. ✅ 调整 `data_root` 和 `csv_root` 路径（如果不在 `data/` 下）
3. ✅ 运行验证工具确认
4. ✅ 运行评估

不需要移动或重命名任何文件！
