# SUScape Dataset Open-Loop Evaluation Guide

This guide explains how to evaluate the ORION model on the SUScape QA dataset for L2 metrics and collision rate in open-loop mode.

## Quick Start

### 1. Prepare Your Dataset

Organize your SUScape dataset as follows:

```
data/suscape_scenes/
└── raws/
    ├── scene-000000/
    │   ├── 0.csv                    # Vehicle trajectory CSV
    │   ├── CAM_FRONT/              # Front camera images
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   └── CAM_BACK_RIGHT/
    ├── scene-000001/
    └── ...
```

### 2. CSV Format

The `0.csv` file should be tab-separated with these columns:
```
TIMESTAMP	TRACK_ID	OBJECT_TYPE	X	Y	V_X	V_Y	A_X	A_Y	YAW	DYAW	DDYAW	CITY_NAME
```

### 3. Run Evaluation

```bash
cd /path/to/Orion_suscape

# First run will generate annotations automatically
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

## What Was Modified

### New Files Created

1. **`mmcv/datasets/suscape_orion_dataset.py`** - Main dataset class
   - Parses CSV trajectory files
   - Loads multi-view camera images
   - Converts world coordinates to ego coordinates
   - Extracts historical and future trajectories

2. **`adzoo/orion/configs/suscape_eval.py`** - Evaluation configuration
   - Dataset paths and settings
   - Object class mapping
   - Evaluation parameters

3. **`docs/SUSCAPE_EVALUATION.md`** - Detailed Chinese tutorial
4. **`docs/SUSCAPE_EVALUATION_EN.md`** - This English guide

### Modified Files

1. **`mmcv/datasets/__init__.py`** - Registered the new SUScape dataset class

## Understanding the Metrics

### L2 Metrics (plan_L2_Xs)
- **Definition**: Average L2 distance between predicted and ground truth trajectories
- **Unit**: meters
- **Lower is better**: Indicates more accurate prediction

### Collision Rate (plan_obj_col_Xs)
- **Definition**: Ratio of predicted trajectories that collide with obstacles
- **Range**: 0-1
- **Lower is better**: Indicates safer planning

### Example Output

```
-------------- Planning Metrics --------------
plan_L2_1s: 0.456        # L2 error at 1 second
plan_L2_2s: 0.892        # L2 error at 2 seconds
plan_L2_3s: 1.234        # L2 error at 3 seconds
plan_obj_col_1s: 0.012   # Collision rate at 1s
plan_obj_col_2s: 0.024   # Collision rate at 2s
plan_obj_col_3s: 0.035   # Collision rate at 3s
```

## Key Implementation Details

### 1. CSV Parsing

The `_parse_csv_file()` method reads the CSV and groups data by timestamp:

```python
df = pd.read_csv(csv_file, sep='\t')
grouped = df.groupby('TIMESTAMP')
```

### 2. Coordinate Transformation

Converts from world coordinates (in CSV) to ego-vehicle coordinates:

```python
# Transform position from world to ego frame
dx, dy = x - ego_x, y - ego_y
cos_ego, sin_ego = np.cos(-ego_yaw), np.sin(-ego_yaw)
x_ego = cos_ego * dx - sin_ego * dy
y_ego = sin_ego * dx + cos_ego * dy
```

### 3. Trajectory Extraction

Extracts past and future trajectories relative to current frame:

```python
def get_ego_trajs(self, index, sample_interval, past_frames, future_frames):
    # Get historical positions
    for i in range(past_frames):
        hist_idx = index - (past_frames - i) * sample_interval
        # Transform to ego frame...
    
    # Get future positions
    for i in range(future_frames):
        fut_idx = index + (i + 1) * sample_interval
        # Transform to ego frame...
```

### 4. Automatic Annotation Generation

If annotation file doesn't exist, it's automatically generated:

```python
def load_annotations(self, ann_file):
    if osp.exists(ann_file):
        return super().load_annotations(ann_file)
    
    # Generate from SUScape raw data
    data_infos = self._generate_annotations_from_suscape()
    mmcv.dump(data_infos, ann_file)
    return data_infos
```

## Customization

### Adjust Detection Range

Edit `suscape_eval.py`:

```python
eval_cfg = {
    "class_range": {
        'car': (60, 60),        # Extend to 60 meters
        'pedestrian': (50, 50),
    }
}
```

### Change Prediction Horizon

Edit `suscape_eval.py`:

```python
past_frames = 3      # Use 3 historical frames
future_frames = 12   # Predict 6 seconds (at 2Hz)
```

### Add New Object Classes

1. Update name mapping:
```python
NameMapping = {
    'Vehicle': 'car',
    'Motorcycle': 'motorcycle',  # New class
    # ...
}
```

2. Update class list:
```python
class_names = ['car', 'motorcycle', 'pedestrian', 'bicycle', 'others']
```

## Troubleshooting

### CSV Format Error

**Problem**: `pandas.errors.ParserError`

**Solution**: Ensure CSV uses tab separation (`\t`) and has consistent columns

### Missing Images

**Problem**: `FileNotFoundError`

**Solution**: Check that camera directories exist and images are named sequentially

### Out of Memory

**Problem**: `CUDA out of memory`

**Solutions**:
- Set `batch_size=1` in config
- Use fewer workers: `workers_per_gpu=2`
- Consider FP16 inference

### Regenerate Annotations

```bash
rm data/suscape_infos/suscape_infos_test.pkl
./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1
```

## How It Works

### Data Flow

1. **Load CSV** → Parse trajectory data for all objects
2. **Load Images** → Load 6-view camera images
3. **Transform Coords** → Convert world coords to ego frame
4. **Extract Trajectories** → Get past/future ego trajectories
5. **Run Model** → Generate predictions
6. **Compute Metrics** → Calculate L2 error and collision rate

### Coordinate Systems

- **World Frame**: CSV data (X, Y, YAW)
- **Ego Frame**: Vehicle-centered (forward = +X, left = +Y)
- **Camera Frame**: Individual camera perspectives
- **Image Frame**: 2D image pixels

### Metric Computation

**L2 Distance**:
```python
L2 = mean(sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2))
```

**Collision Detection**:
1. Project obstacles to BEV grid (200x200, 0.5m resolution)
2. Compute vehicle bounding box positions
3. Check for overlap with obstacle grid

## Technical Notes

- Default camera intrinsics are used (can be customized)
- Default camera extrinsics assume standard positions
- Ego vehicle size: 4.084m (length) × 1.85m (width)
- BEV grid: 100m × 100m with 0.5m resolution
- Supports variable number of objects per frame
- Handles missing future frames gracefully

## References

- ORION Paper: https://arxiv.org/abs/2503.19755
- Project Page: https://xiaomi-mlab.github.io/Orion/
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive

## Support

For issues or questions:
1. Check the detailed Chinese tutorial: `docs/SUSCAPE_EVALUATION.md`
2. Review the dataset class: `mmcv/datasets/suscape_orion_dataset.py`
3. Check configuration: `adzoo/orion/configs/suscape_eval.py`
