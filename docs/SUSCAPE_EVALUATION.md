# SUScape数据集开环评估教程

本文档详细说明如何使用SUScape QA数据集对ORION模型进行开环评估，包括L2指标和碰撞率的计算。

## 目录

1. [数据集准备](#数据集准备)
2. [代码修改说明](#代码修改说明)
3. [配置文件说明](#配置文件说明)
4. [运行评估](#运行评估)
5. [输出结果解读](#输出结果解读)
6. [常见问题](#常见问题)

## 数据集准备

### 1. SUScape数据集结构

确保您的SUScape数据集具有以下结构：

```
data/suscape_scenes/
├── raws/
│   ├── scene-000000/
│   │   ├── 0.csv                    # 车辆轨迹数据
│   │   ├── CAM_FRONT/              # 前视相机图像
│   │   │   ├── 0000.jpg
│   │   │   ├── 0001.jpg
│   │   │   └── ...
│   │   ├── CAM_FRONT_LEFT/         # 左前相机图像
│   │   ├── CAM_FRONT_RIGHT/        # 右前相机图像
│   │   ├── CAM_BACK/               # 后视相机图像
│   │   ├── CAM_BACK_LEFT/          # 左后相机图像
│   │   └── CAM_BACK_RIGHT/         # 右后相机图像
│   ├── scene-000001/
│   └── ...
```

### 2. CSV文件格式

`0.csv` 文件应包含以下列（用制表符分隔）：

```
TIMESTAMP	TRACK_ID	OBJECT_TYPE	X	Y	V_X	V_Y	A_X	A_Y	YAW	DYAW	DDYAW	CITY_NAME
```

- **TIMESTAMP**: 时间戳
- **TRACK_ID**: 轨迹ID（"ego"表示自车）
- **OBJECT_TYPE**: 对象类型（Vehicle, Pedestrian, Bicycle等）
- **X, Y**: 世界坐标系下的位置
- **V_X, V_Y**: 速度分量
- **A_X, A_Y**: 加速度分量
- **YAW**: 航向角（度）
- **DYAW**: 航向角变化率
- **DDYAW**: 航向角加速度
- **CITY_NAME**: 城市名称

### 3. 创建数据目录

```bash
cd /path/to/Orion_suscape
mkdir -p data/suscape_scenes/raws
mkdir -p data/suscape_infos
```

将您的SUScape场景数据放入 `data/suscape_scenes/raws/` 目录中。

## 代码修改说明

我们新增了以下文件来支持SUScape数据集：

### 1. SUScape数据集类

**文件**: `mmcv/datasets/suscape_orion_dataset.py`

这个新增的数据集类主要实现了：

- **CSV解析器**: 读取`0.csv`文件并提取车辆轨迹数据
- **多视角图像加载**: 处理6个相机视角的图像
- **坐标系转换**: 将世界坐标转换为自车坐标系
- **轨迹提取**: 提取历史和未来轨迹用于评估
- **自动标注生成**: 如果标注文件不存在，自动从CSV生成

关键方法：
- `_parse_csv_file()`: 解析CSV文件
- `_build_frame_info()`: 构建每一帧的信息
- `get_ego_trajs()`: 获取自车的历史和未来轨迹
- `get_data_info()`: 准备模型输入数据

### 2. 配置文件

**文件**: `adzoo/orion/configs/suscape_eval.py`

配置文件定义了：
- 数据集路径
- 对象类别映射
- 评估参数
- 模型配置

### 3. 注册数据集

**修改文件**: `mmcv/datasets/__init__.py`

添加了SUScape数据集的导入和注册。

## 配置文件说明

### 关键配置项

#### 1. 数据路径配置

```python
dataset_type = "SUScapeOrionDataset"
data_root = "data/suscape_scenes"      # SUScape数据集根目录
info_root = "data/suscape_infos"       # 标注文件保存目录
ann_file_test = info_root + "/suscape_infos_test.pkl"
```

#### 2. 类别映射

```python
NameMapping = {
    'Vehicle': 'car',
    'Pedestrian': 'pedestrian',
    'Bicycle': 'bicycle',
}

class_names = ['car', 'pedestrian', 'bicycle', 'others']
```

#### 3. 评估配置

```python
eval_cfg = {
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "class_names": ['car', 'pedestrian', 'bicycle'],
    "class_range": {
        'car': (50, 50),           # 检测范围 (前后, 左右) 米
        'pedestrian': (40, 40),
        'bicycle': (40, 40)
    }
}
```

#### 4. 轨迹预测配置

```python
past_frames = 2      # 历史帧数
future_frames = 6    # 未来帧数（对应3秒，假设2Hz采样）
```

## 运行评估

### 1. 准备模型权重

确保您已下载ORION模型权重：

```bash
mkdir -p ckpts
cd ckpts
# 下载预训练权重
wget https://huggingface.co/poleyzdk/Orion/resolve/main/Orion.pth
```

### 2. 首次运行（生成标注）

第一次运行时，系统会自动扫描SUScape数据并生成标注文件：

```bash
cd /path/to/Orion_suscape
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

这个过程会：
1. 扫描 `data/suscape_scenes/raws/` 目录中的所有场景
2. 解析每个场景的 `0.csv` 文件
3. 生成标注文件并保存到 `data/suscape_infos/suscape_infos_test.pkl`
4. 开始评估

### 3. 后续运行

标注文件生成后，后续运行会直接加载标注：

```bash
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

### 4. 多GPU评估

如果有多个GPU，可以使用：

```bash
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 4  # 使用4个GPU
```

## 输出结果解读

### 评估指标

运行完成后，您会看到以下规划指标输出：

```
-------------- Planning Metrics --------------
plan_L2_1s: 0.456        # 1秒L2误差（米）
plan_L2_2s: 0.892        # 2秒L2误差（米）
plan_L2_3s: 1.234        # 3秒L2误差（米）
plan_obj_col_1s: 0.012   # 1秒目标碰撞率
plan_obj_col_2s: 0.024   # 2秒目标碰撞率
plan_obj_col_3s: 0.035   # 3秒目标碰撞率
plan_obj_box_col_1s: 0.008   # 1秒边界框碰撞率
plan_obj_box_col_2s: 0.015   # 2秒边界框碰撞率
plan_obj_box_col_3s: 0.022   # 3秒边界框碰撞率
fut_valid_flag: True     # 有效样本标志
```

### 指标含义

#### L2指标 (plan_L2_Xs)
- **定义**: 预测轨迹与真实轨迹之间的平均L2距离（欧几里得距离）
- **单位**: 米
- **计算**: 对于X秒（1s/2s/3s），计算预测轨迹点与对应真实轨迹点之间的平均距离
- **越低越好**: 表示预测越准确

#### 目标碰撞率 (plan_obj_col_Xs)
- **定义**: 预测轨迹与障碍物发生碰撞的比率
- **范围**: 0-1
- **计算**: 检查预测轨迹中心点是否与占据栅格重叠
- **越低越好**: 表示规划越安全

#### 边界框碰撞率 (plan_obj_box_col_Xs)
- **定义**: 考虑车辆边界框的碰撞率
- **范围**: 0-1
- **计算**: 使用完整车辆边界框检查与障碍物的碰撞
- **越低越好**: 更保守的碰撞检测

## 常见问题

### Q1: CSV文件格式错误

**错误**: `pandas.errors.ParserError: Error tokenizing data`

**解决方案**:
- 检查CSV文件是否使用制表符（`\t`）分隔
- 确保所有行的列数相同
- 检查是否有特殊字符或编码问题

### Q2: 图像文件未找到

**错误**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决方案**:
- 确保每个场景目录下有对应的相机目录
- 检查图像文件命名是否正确
- 验证图像文件格式（.jpg或.png）

### Q3: 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小batch_size（在配置文件中设置为1）
- 使用FP16推理（创建suscape_eval_fp16.py配置）
- 减少workers_per_gpu数量

### Q4: 标注文件需要重新生成

如果需要重新生成标注文件：

```bash
rm data/suscape_infos/suscape_infos_test.pkl
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

### Q5: 相机参数不准确

如果您知道准确的相机参数，可以修改 `suscape_orion_dataset.py` 中的：
- `_get_default_intrinsic()`: 修改内参矩阵
- `_get_default_cam2ego()`: 修改相机外参

### Q6: 坐标系转换问题

SUScape使用的坐标系统：
- **世界坐标系**: CSV文件中的X, Y, YAW
- **自车坐标系**: 以自车为原点，前方为X轴正方向
- **LiDAR坐标系**: 与自车坐标系对齐

代码会自动处理这些转换。

## 高级用法

### 自定义评估范围

修改配置文件中的 `eval_cfg`:

```python
eval_cfg = {
    "class_range": {
        'car': (60, 60),        # 扩大到60米
        'pedestrian': (50, 50),
    }
}
```

### 调整轨迹预测时长

修改配置文件：

```python
past_frames = 3      # 使用3帧历史
future_frames = 12   # 预测6秒（假设2Hz）
```

### 添加新的对象类别

1. 修改 `NameMapping`:
```python
NameMapping = {
    'Vehicle': 'car',
    'Pedestrian': 'pedestrian',
    'Bicycle': 'bicycle',
    'Motorcycle': 'motorcycle',  # 新增
}
```

2. 修改 `class_names`:
```python
class_names = ['car', 'pedestrian', 'bicycle', 'motorcycle', 'others']
```

3. 相应修改模型配置中的 `num_classes`

## 技术细节

### L2指标计算

L2误差计算采用与STP3相同的方法：

```python
def compute_L2(trajs, gt_trajs):
    pred_len = trajs.shape[0]
    ade = sum(
        sqrt((trajs[i,0] - gt_trajs[i,0])^2 + (trajs[i,1] - gt_trajs[i,1])^2)
        for i in range(pred_len)
    ) / pred_len
    return ade
```

### 碰撞检测

使用BEV占据栅格进行碰撞检测：
1. 将障碍物投影到BEV栅格（200x200，分辨率0.5m）
2. 计算自车边界框在栅格中的位置
3. 检查是否与障碍物栅格重叠

### 坐标系转换流程

```
世界坐标 (CSV) 
    ↓ (world2lidar矩阵)
LiDAR坐标 (自车中心)
    ↓ (lidar2cam矩阵)
相机坐标
    ↓ (内参矩阵)
图像坐标
```

## 参考资料

- ORION论文: https://arxiv.org/abs/2503.19755
- Bench2Drive评估: https://github.com/Thinklab-SJTU/Bench2Drive
- STP3指标: https://github.com/OpenDriveLab/ST-P3

## 更新日志

- 2025-01-17: 初始版本，支持SUScape数据集的L2和碰撞率评估
