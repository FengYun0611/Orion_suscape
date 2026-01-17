# How to Evaluate ORION on SUScape Dataset

## 修改说明 / What Was Changed

本项目已完成对 SUScape QA 数据集的完整支持，您现在可以直接使用 SUScape 数据集对 ORION 模型进行**开环评估**，包括 **L2 指标**和**碰撞率**的计算。

This project now fully supports the SUScape QA dataset for **open-loop evaluation** of the ORION model, including **L2 metrics** and **collision rate** calculations.

## 快速开始 / Quick Start

### 第一步：准备数据 / Step 1: Prepare Data

确保您的 SUScape 数据集按以下结构组织：

Ensure your SUScape dataset is organized as follows:

```
data/suscape_scenes/
└── raws/
    ├── scene-000000/
    │   ├── 0.csv                    # 轨迹数据 / Trajectory data
    │   ├── CAM_FRONT/              # 前视相机 / Front camera
    │   ├── CAM_FRONT_LEFT/         # 左前相机 / Front-left camera  
    │   ├── CAM_FRONT_RIGHT/        # 右前相机 / Front-right camera
    │   ├── CAM_BACK/               # 后视相机 / Back camera
    │   ├── CAM_BACK_LEFT/          # 左后相机 / Back-left camera
    │   └── CAM_BACK_RIGHT/         # 右后相机 / Back-right camera
    ├── scene-000001/
    └── ...
```

**CSV 文件格式 / CSV File Format** (`0.csv` - tab-separated):
```
TIMESTAMP	TRACK_ID	OBJECT_TYPE	X	Y	V_X	V_Y	A_X	A_Y	YAW	DYAW	DDYAW	CITY_NAME
```

### 第二步：验证数据集 / Step 2: Verify Dataset

运行验证工具检查数据集结构：

Run the verification tool to check dataset structure:

```bash
python tools/verify_suscape_dataset.py data/suscape_scenes
```

### 第三步：运行评估 / Step 3: Run Evaluation

```bash
# 单 GPU 评估 / Single GPU evaluation
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1

# 多 GPU 评估 / Multi-GPU evaluation (e.g., 4 GPUs)
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 4
```

**首次运行 / First Run:**
- 系统会自动扫描 CSV 文件生成标注 / System automatically generates annotations from CSV files
- 标注保存在 `data/suscape_infos/suscape_infos_test.pkl` / Annotations saved to `data/suscape_infos/suscape_infos_test.pkl`

**后续运行 / Subsequent Runs:**
- 直接加载已生成的标注文件 / Directly loads pre-generated annotation file

### 第四步：查看结果 / Step 4: View Results

评估完成后会输出以下指标：

After evaluation, the following metrics will be displayed:

```
-------------- Planning Metrics --------------
plan_L2_1s: 0.456        # L2 误差 @ 1秒 / L2 error @ 1 second (meters)
plan_L2_2s: 0.892        # L2 误差 @ 2秒 / L2 error @ 2 seconds (meters)
plan_L2_3s: 1.234        # L2 误差 @ 3秒 / L2 error @ 3 seconds (meters)
plan_obj_col_1s: 0.012   # 碰撞率 @ 1秒 / Collision rate @ 1 second
plan_obj_col_2s: 0.024   # 碰撞率 @ 2秒 / Collision rate @ 2 seconds
plan_obj_col_3s: 0.035   # 碰撞率 @ 3秒 / Collision rate @ 3 seconds
plan_obj_box_col_1s: 0.008   # 边界框碰撞率 @ 1秒 / Box collision rate @ 1s
plan_obj_box_col_2s: 0.015   # 边界框碰撞率 @ 2秒 / Box collision rate @ 2s
plan_obj_box_col_3s: 0.022   # 边界框碰撞率 @ 3秒 / Box collision rate @ 3s
```

**指标含义 / Metric Definitions:**
- **L2**: 预测轨迹与真实轨迹的平均欧几里得距离（越低越好）/ Average Euclidean distance between predicted and ground truth trajectories (lower is better)
- **Collision Rate**: 预测轨迹与障碍物发生碰撞的比率（越低越好）/ Ratio of predicted trajectories that collide with obstacles (lower is better)

## 新增文件 / New Files

### 核心代码 / Core Code

1. **`mmcv/datasets/suscape_orion_dataset.py`**
   - SUScape 数据集类 / SUScape dataset class
   - CSV 解析和坐标转换 / CSV parsing and coordinate transformation
   - 自动标注生成 / Automatic annotation generation

2. **`adzoo/orion/configs/suscape_eval.py`**
   - SUScape 评估配置文件 / SUScape evaluation configuration
   - 包含数据路径、类别映射、评估参数 / Includes data paths, class mapping, evaluation parameters

### 工具 / Tools

3. **`tools/verify_suscape_dataset.py`**
   - 数据集结构验证工具 / Dataset structure verification tool
   - CSV 格式检查 / CSV format validation

### 文档 / Documentation

4. **`docs/SUSCAPE_EVALUATION.md`** - 中文详细教程 / Detailed Chinese tutorial
5. **`docs/SUSCAPE_EVALUATION_EN.md`** - English tutorial
6. **`docs/SUSCAPE_QUICKSTART.md`** - 快速参考 / Quick reference
7. **`docs/IMPLEMENTATION_SUMMARY.md`** - 实现总结 / Implementation summary
8. **`docs/HOW_TO_USE_SUSCAPE.md`** - 本文件 / This file

## 修改文件 / Modified Files

- **`mmcv/datasets/__init__.py`**: 注册 SUScape 数据集 / Registered SUScape dataset

## 技术要点 / Technical Highlights

### 1. CSV 解析 / CSV Parsing

```python
# Tab-separated CSV with 13 columns
df = pd.read_csv('0.csv', sep='\t')
grouped = df.groupby('TIMESTAMP')  # Group by timestamp
```

### 2. 坐标转换 / Coordinate Transformation

```python
# World coordinates → Ego-vehicle coordinates
dx, dy = x - ego_x, y - ego_y
cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
x_ego = cos_yaw * dx - sin_yaw * dy
y_ego = sin_yaw * dx + cos_yaw * dy
```

### 3. 轨迹提取 / Trajectory Extraction

- **历史 / History**: `past_frames=2` (2 frames)
- **未来 / Future**: `future_frames=6` (6 frames = 3 seconds @ 2Hz)

### 4. 评估计算 / Metric Computation

**L2 Distance:**
```python
L2 = mean(sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2))
```

**Collision Detection:**
- BEV 栅格 / BEV grid: 200×200, 分辨率 / resolution: 0.5m
- 检查车辆边界框与障碍物重叠 / Check vehicle bounding box overlap with obstacles

## 自定义配置 / Customization

### 修改检测范围 / Modify Detection Range

编辑 / Edit `adzoo/orion/configs/suscape_eval.py`:

```python
eval_cfg = {
    "class_range": {
        'car': (60, 60),        # 前后左右 60 米 / 60m in all directions
        'pedestrian': (50, 50),
    }
}
```

### 调整预测时长 / Adjust Prediction Horizon

```python
past_frames = 3      # 3 帧历史 / 3 historical frames
future_frames = 12   # 预测 6 秒 / Predict 6 seconds (@ 2Hz)
```

### 添加新类别 / Add New Classes

```python
NameMapping = {
    'Vehicle': 'car',
    'Motorcycle': 'motorcycle',  # 新增 / New class
    'Pedestrian': 'pedestrian',
    'Bicycle': 'bicycle',
}
```

## 常见问题 / FAQ

### Q1: CSV 格式错误 / CSV Format Error

**问题 / Problem**: `pandas.errors.ParserError`

**解决方案 / Solution**:
- 确保使用制表符分隔 / Ensure tab-separated (`\t`)
- 检查列名是否完整 / Check all column names are present
- 确保有 `TRACK_ID='ego'` 的数据 / Ensure data with `TRACK_ID='ego'` exists

### Q2: 找不到图像 / Images Not Found

**问题 / Problem**: `FileNotFoundError`

**解决方案 / Solution**:
- 检查相机目录是否存在 / Check camera directories exist
- 确认图像格式为 `.jpg` 或 `.png` / Confirm images are `.jpg` or `.png`
- 确保图像按顺序命名 / Ensure images are named sequentially

### Q3: 内存不足 / Out of Memory

**问题 / Problem**: `CUDA out of memory`

**解决方案 / Solution**:
- 设置 `batch_size=1` / Set `batch_size=1` in config
- 减少 `workers_per_gpu` / Reduce `workers_per_gpu`
- 考虑使用 FP16 推理 / Consider FP16 inference

### Q4: 重新生成标注 / Regenerate Annotations

```bash
rm data/suscape_infos/suscape_infos_test.pkl
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

## 数据流程 / Data Flow

```
CSV 文件 / CSV Files
    ↓ (解析 / Parse)
世界坐标 / World Coordinates
    ↓ (转换 / Transform)
自车坐标 / Ego Coordinates
    ↓ (提取轨迹 / Extract Trajectories)
历史+未来轨迹 / Past + Future Trajectories
    ↓ (模型推理 / Model Inference)
预测轨迹 / Predicted Trajectories
    ↓ (计算指标 / Compute Metrics)
L2 & 碰撞率 / L2 & Collision Rate
```

## 完整示例 / Complete Example

```bash
# 1. 准备数据 / Prepare data
# 将 SUScape 数据放在 data/suscape_scenes/raws/
# Place SUScape data in data/suscape_scenes/raws/

# 2. 验证数据集 / Verify dataset
python tools/verify_suscape_dataset.py data/suscape_scenes

# 3. 运行评估 / Run evaluation
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1

# 4. 查看输出 / View output
# 检查终端输出的评估指标 / Check evaluation metrics in terminal output
```

## 更多帮助 / More Help

- **中文详细教程 / Detailed Chinese Tutorial**: `docs/SUSCAPE_EVALUATION.md`
- **English Tutorial**: `docs/SUSCAPE_EVALUATION_EN.md`
- **快速参考 / Quick Reference**: `docs/SUSCAPE_QUICKSTART.md`
- **实现细节 / Implementation Details**: `docs/IMPLEMENTATION_SUMMARY.md`

## 参考 / References

- ORION Paper: https://arxiv.org/abs/2503.19755
- Project Page: https://xiaomi-mlab.github.io/Orion/
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
