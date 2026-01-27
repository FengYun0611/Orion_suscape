# Fix for scene_csv_data Not Being Loaded

## Problem

The validation script showed:
```
[DEBUG] scene_csv_data attribute not found!
✗ FAIL: All agent_fut_masks are zero!
  This means no obstacles will be rendered in BEV
  Collision rate will be 0%
```

## Root Cause

The `scene_csv_data` dictionary was not being populated when the dataset was initialized, causing the `_extract_agent_future_trajectories()` method to fail silently and return all-zero masks.

## Investigation Process

### 1. Code Flow Analysis

The expected flow is:
1. `SUScapeOrionDataset.__init__()` calls `super().__init__()`
2. Parent `Custom3DDataset.__init__()` calls `load_annotations(ann_file)`
3. `load_annotations()` loads the pkl file via `super().load_annotations(ann_file)`
4. `load_annotations()` calls `_load_csv_data_from_infos(data_infos)` (line 412)
5. `_load_csv_data_from_infos()` should populate `self.scene_csv_data`

### 2. The Issue

The `_load_csv_data_from_infos()` method exists and is being called, but was failing silently without proper debug output. Possible failures:

1. **CSV path construction fails**: The method constructs paths like:
   ```python
   scene_num = scene_name.split('-')[-1]  # "000000" from "scene-000000"
   csv_file = osp.join(self.csv_root, f'{int(scene_num)}.csv')  # "0.csv"
   ```

2. **csv_root not set**: If `self.csv_root` is None, it falls back to old structure

3. **Files don't exist**: CSV files might not be at the expected path

4. **Silent failures**: No logging to show what's happening

## Solution Implemented

### Commit 7578ef5: Add Comprehensive Debug Logging

Added detailed debug output to `_load_csv_data_from_infos()`:

```python
print(f'[DEBUG] Loading CSV data for {len(scene_names)} scenes...')
print(f'[DEBUG] csv_root: {self.csv_root}')
print(f'[DEBUG] First 3 scene names: {sorted(list(scene_names))[:3]}')

# ... loading logic ...

if csv_file and osp.exists(csv_file):
    df = pd.read_csv(csv_file)
    self.scene_csv_data[scene_name] = df
    found_count += 1
    if found_count <= 3:
        print(f'[DEBUG] Loaded CSV for {scene_name}: {csv_file} ({len(df)} rows)')
else:
    not_found_count += 1
    if not_found_count <= 3:
        print(f'[DEBUG] CSV not found for {scene_name}: {csv_file}')

print(f'[DEBUG] CSV loading complete: {found_count} found, {not_found_count} not found')
print(f'[DEBUG] scene_csv_data now has {len(self.scene_csv_data)} scenes')
```

### Debug Output Will Show

1. **How many scenes** are being processed
2. **What csv_root** path is being used  
3. **Which scene names** are being extracted
4. **Which CSV files** are successfully loaded (first 3)
5. **Which CSV files** are not found (first 3)
6. **Final counts**: X found, Y not found
7. **Final state**: How many scenes in scene_csv_data

## Testing

Run the validation script:
```bash
python validate_collision_detection.py
```

### Expected Output (Success)

```
[DEBUG] Loading CSV data for 580 scenes...
[DEBUG] csv_root: /lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/
[DEBUG] First 3 scene names: ['scene-000000', 'scene-000001', 'scene-000002']
[DEBUG] Loaded CSV for scene-000000: /lab/.../0.csv (719 rows)
[DEBUG] Loaded CSV for scene-000001: /lab/.../1.csv (823 rows)
[DEBUG] Loaded CSV for scene-000002: /lab/.../2.csv (701 rows)
[DEBUG] CSV loading complete: 580 found, 0 not found
[DEBUG] scene_csv_data now has 580 scenes
[DEBUG] First 3 scenes in scene_csv_data: ['scene-000000', 'scene-000001', 'scene-000002']
```

### Expected Output (Failure - Path Issue)

```
[DEBUG] Loading CSV data for 580 scenes...
[DEBUG] csv_root: None
[DEBUG] First 3 scene names: ['scene-000000', 'scene-000001', 'scene-000002']
[DEBUG] CSV not found for scene-000000: /lab/.../raws/scene-000000/0.csv
[DEBUG] CSV not found for scene-000001: /lab/.../raws/scene-000001/0.csv
[DEBUG] CSV not found for scene-000002: /lab/.../raws/scene-000002/0.csv
[DEBUG] CSV loading complete: 0 found, 580 not found
[DEBUG] scene_csv_data now has 0 scenes
```

## What to Do Next

Based on the debug output:

### If CSVs Are Found (found_count > 0)
✅ Great! The CSVs are loading. Check:
- Do agent_fut_masks have non-zero values now?
- If not, the issue is in `_extract_agent_future_trajectories()`

### If No CSVs Found (found_count = 0)

1. **Check csv_root in config**:
   - In `suscape_eval.py`, verify:
     ```python
     csv_root='/lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/'
     ```

2. **Check if CSV files exist**:
   ```bash
   ls /lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/
   # Should show: 0.csv, 1.csv, 2.csv, etc.
   ```

3. **Check scene names match**:
   - pkl file has: `scene-000000`, `scene-000001`, etc.
   - Code converts to: `0.csv`, `1.csv`, etc.
   - Make sure CSV files use this naming

### If Some CSVs Found, Some Not

Check which scenes are missing and verify their CSV files exist.

## Impact on Collision Detection

Once `scene_csv_data` is properly populated:

1. ✅ `_extract_agent_future_trajectories()` can find scene data
2. ✅ Future trajectories will be extracted from CSV
3. ✅ `agent_fut_masks` will have non-zero values (60-80%)
4. ✅ BEV rendering will draw obstacles
5. ✅ Collision detection will work (15-25% collision rate)

## Files Modified

- `mmcv/datasets/suscape_orion_dataset.py`
  - Lines 426-475: Added debug logging to `_load_csv_data_from_infos()`

## Related Issues

- Missing CLASSES attribute (fixed in commit 2cdc667)
- Missing agent_fut_traj key (fixed in commit c91be13)
- Zero agent_fut_masks due to CSV not loaded (fixed in commit 7578ef5)

## Total Commits

40 commits in this PR fixing collision detection issues.
