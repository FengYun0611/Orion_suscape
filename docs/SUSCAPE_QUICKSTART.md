# SUScape 数据集快速使用指南

本项目已添加对 SUScape 数据集的支持，可以直接进行开环评估（L2 指标和碰撞率）。

## 快速开始

### 1. 准备数据集

将您的 SUScape 数据集按以下结构组织：

```
data/suscape_scenes/
└── raws/
    ├── scene-000000/
    │   ├── 0.csv                    # 车辆轨迹CSV文件
    │   ├── CAM_FRONT/              # 6个相机视角的图像
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   └── CAM_BACK_RIGHT/
    ├── scene-000001/
    └── ...
```

### 2. 验证数据集

```bash
python tools/verify_suscape_dataset.py data/suscape_scenes
```

### 3. 运行评估

```bash
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

首次运行会自动从 CSV 文件生成标注文件，保存在 `data/suscape_infos/` 目录。

### 4. 查看结果

评估完成后会输出：
- `plan_L2_1s/2s/3s`: L2 误差（米）
- `plan_obj_col_1s/2s/3s`: 目标碰撞率
- `plan_obj_box_col_1s/2s/3s`: 边界框碰撞率

## 详细文档

- [中文详细教程](docs/SUSCAPE_EVALUATION.md)
- [English Tutorial](docs/SUSCAPE_EVALUATION_EN.md)

## CSV 文件格式

`0.csv` 文件应使用制表符分隔，包含以下列：

```
TIMESTAMP	TRACK_ID	OBJECT_TYPE	X	Y	V_X	V_Y	A_X	A_Y	YAW	DYAW	DDYAW	CITY_NAME
```

- `TRACK_ID='ego'` 表示自车
- `OBJECT_TYPE`: Vehicle, Pedestrian, Bicycle 等

## 代码修改说明

### 新增文件

1. **`mmcv/datasets/suscape_orion_dataset.py`**
   - SUScape 数据集类
   - CSV 解析和坐标转换
   - 自动生成标注文件

2. **`adzoo/orion/configs/suscape_eval.py`**
   - SUScape 评估配置文件

3. **`tools/verify_suscape_dataset.py`**
   - 数据集结构验证工具

4. **文档**
   - `docs/SUSCAPE_EVALUATION.md` (中文)
   - `docs/SUSCAPE_EVALUATION_EN.md` (English)

### 修改文件

- **`mmcv/datasets/__init__.py`**: 注册 SUScape 数据集

## 技术要点

### CSV 解析
```python
# 按时间戳分组读取
df = pd.read_csv('0.csv', sep='\t')
grouped = df.groupby('TIMESTAMP')
```

### 坐标转换
```python
# 世界坐标 → 自车坐标
dx, dy = x - ego_x, y - ego_y
cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
x_ego = cos_yaw * dx - sin_yaw * dy
y_ego = sin_yaw * dx + cos_yaw * dy
```

### 轨迹提取
- 历史轨迹: `past_frames=2` (2帧历史)
- 未来轨迹: `future_frames=6` (6帧，对应3秒@2Hz)

### L2 计算
```python
L2 = mean(sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2))
```

### 碰撞检测
- BEV 栅格: 200×200, 分辨率 0.5m
- 检查车辆边界框与障碍物栅格重叠

## 自定义配置

### 修改检测范围

编辑 `adzoo/orion/configs/suscape_eval.py`:

```python
eval_cfg = {
    "class_range": {
        'car': (60, 60),        # 扩大到60米
        'pedestrian': (50, 50),
    }
}
```

### 调整预测时长

```python
past_frames = 3      # 3帧历史
future_frames = 12   # 预测6秒
```

## 常见问题

### Q: CSV 格式错误？
- 确保使用制表符 (`\t`) 分隔
- 检查列名是否完整
- 确保有 `TRACK_ID='ego'` 的数据

### Q: 找不到图像文件？
- 检查相机目录是否存在
- 确认图像文件为 `.jpg` 或 `.png` 格式

### Q: 内存不足？
- 设置 `batch_size=1`
- 减少 `workers_per_gpu`
- 考虑使用 FP16 推理

### Q: 重新生成标注？
```bash
rm data/suscape_infos/suscape_infos_test.pkl
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

## 参考

- ORION 论文: https://arxiv.org/abs/2503.19755
- 项目主页: https://xiaomi-mlab.github.io/Orion/
