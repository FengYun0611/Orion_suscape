# 用户数据结构适配说明 (User-Specific Setup Guide)

## 您的数据结构 (Your Data Structure)

```
data/
├── suscape_scenes/
│   ├── scene-000000/
│   │   └── camera/
│   │       ├── front/
│   │       │   ├── 1630376940.000.jpg  # base timestamp
│   │       │   ├── 1630376940.500.jpg  # base + 0.5s (frame 1)
│   │       │   ├── 1630376941.000.jpg  # base + 1.0s (frame 2)
│   │       │   └── ...
│   │       ├── front_left/
│   │       ├── front_right/
│   │       ├── rear/
│   │       ├── rear_left/
│   │       └── rear_right/
│   ├── scene-000001/
│   └── ...
│
├── suscape_scene_traj_csv_alldistance_fixyaw/
│   ├── 0.csv      # scene-000000的轨迹数据
│   ├── 1.csv      # scene-000001的轨迹数据
│   └── ...
│
└── sharegpt_dataset/
    ├── dataset_info.json
    ├── suscape_NQA_q7.json   # 端到端轨迹预测
    └── suscape_NQA_qX.json   # 其他QA任务
```

## QA数据格式 (QA Data Format)

### ID格式 (ID Format)
```json
{
  "id": "scene-000000_7_q7",  // scene-000000的第8帧 (从0开始)
  "conversations": [
    {
      "from": "human",
      "value": "You are an autonomous driving agent..."
    },
    {
      "from": "gpt",
      "value": "Predicted future movement... [x, y]: [5.15, 0.02], [10.31, 0.05], ..."
    }
  ]
}
```

### 轨迹格式 (Trajectory Format)
```
[x, y]: [5.15, 0.02], [10.31, 0.05], [15.47, 0.10], ...
```

- **坐标系**: Positive x = 前进, Positive y = 左侧
- **单位**: 米 (meters)
- **采样**: 0.5秒间隔
- **点数**: 通常9个点 (4.5秒), 系统使用前6个 (3秒)

## 图片命名规则 (Image Naming Rule)

**公式**: `filename = (base_timestamp + frame_index * 0.5).jpg`

**示例** (Example):
- scene-000000的0.csv中，第0帧timestamp = 1630376940
- Frame 0: `1630376940.000.jpg`
- Frame 1: `1630376940.500.jpg`
- Frame 7: `1630376943.500.jpg`

保留3位小数！

## 相机名称映射 (Camera Name Mapping)

| 您的目录 (Your Dir) | 系统名称 (System Name) | 说明 |
|-------------------|---------------------|-----|
| `camera/front` | `CAM_FRONT` | 前视 |
| `camera/front_left` | `CAM_FRONT_LEFT` | 左前 |
| `camera/front_right` | `CAM_FRONT_RIGHT` | 右前 |
| `camera/rear` | `CAM_BACK` | 后视 |
| `camera/rear_left` | `CAM_BACK_LEFT` | 左后 |
| `camera/rear_right` | `CAM_BACK_RIGHT` | 右后 |

## 配置文件 (Configuration)

编辑 `adzoo/orion/configs/suscape_eval.py`:

```python
dataset_root = 'data'
data_root = f'{dataset_root}/suscape_scenes'
csv_root = f'{dataset_root}/suscape_scene_traj_csv_alldistance_fixyaw'
qa_root = f'{dataset_root}/sharegpt_dataset'
qa_tasks = ['q7']  # 端到端轨迹预测

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(
        type='SUScapeOrionDataset',
        data_root=data_root,
        csv_root=csv_root,
        qa_root=qa_root,
        qa_tasks=qa_tasks,
        ann_file=f'{data_root}/suscape_infos_val.pkl',
        ...
    )
)
```

## 使用步骤 (Usage Steps)

### 1. 验证数据集 (Verify Dataset)
```bash
python tools/verify_suscape_dataset.py \
    data/suscape_scenes \
    data/suscape_scene_traj_csv_alldistance_fixyaw
```

### 2. 运行评估 (Run Evaluation)
```bash
./adzoo/orion/orion_dist_eval.sh \
    adzoo/orion/configs/suscape_eval.py \
    ckpts/Orion.pth \
    1
```

### 3. 查看结果 (View Results)
评估会输出以下指标:

```
============================================================
Planning Metrics Evaluation (SUScape QA Dataset)
============================================================

Total valid samples: XXXX

L2 Trajectory Error (meters):
  1s: X.XXXX  # 2个路径点 @ 0.5s
  2s: X.XXXX  # 4个路径点
  3s: X.XXXX  # 6个路径点
  Avg: X.XXXX # 平均值

Object Collision Rate:
  1s: X.XXXX
  2s: X.XXXX
  3s: X.XXXX
  Avg: X.XXXX

Bounding Box Collision Rate:
  1s: X.XXXX
  2s: X.XXXX
  3s: X.XXXX
  Avg: X.XXXX
============================================================
```

## 系统工作原理 (How It Works)

### 图片加载逻辑 (Image Loading Logic)
1. 尝试旧格式: `scene-XXXXXX/CAM_FRONT/`
2. 降级到新格式: `scene-XXXXXX/camera/front/`
3. 使用时间戳命名: `{timestamp:.3f}.jpg`
4. 降级到索引: 如果时间戳文件不存在，使用第N个文件

### QA轨迹提取 (QA Trajectory Extraction)
1. 从 `conversations[1]["value"]` 读取GPT回复
2. 使用正则表达式提取 `[x, y]` 坐标对
3. 解析示例: `[5.15, 0.02], [10.31, 0.05], ...`
4. 转换为numpy数组: `[[5.15, 0.02], [10.31, 0.05], ...]`
5. 取前6个点用于3秒预测 (2Hz采样)

### 帧号映射 (Frame Number Mapping)
- QA ID: `scene-000000_7_q7` → 场景: scene-000000, 帧号: 7
- CSV: 按timestamp分组，每个唯一timestamp = 1帧
- 图片: timestamp = base_ts + frame_idx * 0.5

## 特殊处理 (Special Handling)

### 自动降级 (Auto Fallback)
- 如果QA数据不可用 → 使用CSV数据
- 如果时间戳图片不存在 → 使用索引查找
- 如果camera/目录不存在 → 尝试CAM_*目录

### 兼容性 (Compatibility)
- ✅ 支持您的新格式 (camera/front + timestamp.jpg)
- ✅ 兼容旧格式 (CAM_FRONT + index.jpg)
- ✅ 混合使用也可以 (部分场景用旧格式，部分用新格式)

## 常见问题 (FAQ)

### Q: 图片找不到怎么办？
A: 系统会自动尝试:
1. 时间戳命名 (`1630376940.500.jpg`)
2. 索引命名 (第N个文件)
3. 不同目录结构 (camera/ vs CAM_*)

### Q: QA数据格式不对怎么办？
A: 系统支持多种格式:
- `[x, y]: [5.15, 0.02], ...` (主要格式)
- `(5.15, 0.02), ...` (备用格式)
- `x: 5.15, y: 0.02; ...` (备用格式)

### Q: 坐标系转换？
A: QA中的坐标已经是车辆坐标系:
- +x = 前进方向
- +y = 左侧方向
- 无需额外转换

### Q: 如何只用CSV评估？
A: 在配置文件中设置:
```python
qa_root = None
qa_tasks = []
```

## 性能建议 (Performance Tips)

1. **首次运行**: 会自动生成PKL标注文件，需要较长时间
2. **后续运行**: 直接读取PKL文件，速度很快
3. **并行处理**: 调整 `workers_per_gpu` 参数加速
4. **GPU数量**: 使用 `orion_dist_eval.sh` 的第3个参数

---

**完成时间**: 2026-01-18

**提交**: 1682d59

**状态**: ✅ 已测试并验证
