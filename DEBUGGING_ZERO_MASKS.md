# Debugging Zero agent_fut_masks Issue

## Problem
All `agent_fut_masks` are zero (0/120 = 0.00%), which means:
- No obstacles will be rendered in BEV occupancy map
- Collision detection will return 0% collision rate
- Future trajectory extraction is not working

## Root Cause Investigation

The issue is in `_extract_agent_future_trajectories()` function. Possible causes:

### 1. Timestamp Mismatch (Most Likely)
**Symptom**: `current_timestamp` from pkl doesn't match CSV timestamps

**Example**:
- pkl timestamp: `1630376940` (integer or different precision)
- CSV timestamp: `1630376940.0` (float)
- Exact equality check fails: `1630376940 != 1630376940.0` in pandas

**Fix Applied**: Changed to approximate matching with 0.01s tolerance
```python
# Before: exact match
current_idx = all_timestamps.index(current_timestamp)

# After: approximate match
for idx, ts in enumerate(all_timestamps):
    if abs(ts - current_timestamp) < 0.01:
        current_idx = idx
        break
```

### 2. Track ID Mismatch
**Symptom**: `track_id` from `gt_ids` doesn't match CSV `TRACK_ID` column

**Possible Issues**:
- Type mismatch: string vs object in pandas
- Value mismatch: different formats or prefixes

**Check**: Debug output will show:
```
[DEBUG] Agent 0 track_id=scene-000000-1 fut_idx=0 fut_timestamp=1630376940.5: NOT FOUND
[DEBUG]   Tracks at fut_timestamp=1630376940.5: ['scene-000000-2', 'scene-000000-3', ...]
```

### 3. CSV Not Loaded
**Symptom**: `scene_csv_data` doesn't contain the scene

**Check**: Debug output will show:
```
[DEBUG] Scene scene-000000 not in scene_csv_data. Available: ['scene-000001', 'scene-000002']
```

## How to Diagnose

### Step 1: Run Validation Script
```bash
python validate_collision_detection.py
```

### Step 2: Check Debug Output
Look for these debug messages in the output:

**Success indicators:**
```
✓ Agent future trajectories found
  Shape: (20, 12)
  Mask shape: (20, 6)
  Non-zero mask values: 80/120
  Percentage: 66.67%
```

**Failure indicators:**
```
[DEBUG] Current timestamp 1630376940.0 not found in CSV timestamps.
[DEBUG] CSV has 40 timestamps, first 10: [...]
```

or

```
[DEBUG] Agent 0 track_id=scene-000000-1 fut_idx=0: NOT FOUND
[DEBUG]   Tracks at fut_timestamp=1630376940.5: [...]
```

### Step 3: Identify the Issue

Based on debug output:

**If "Current timestamp not found":**
- Issue: Timestamp precision mismatch
- Solution: Increase tolerance or check pkl timestamp format

**If "Agent track_id NOT FOUND":**
- Issue: Track ID doesn't match between pkl and CSV
- Solution: Check track_id format in both sources

**If "Scene not in scene_csv_data":**
- Issue: CSV wasn't loaded for this scene
- Solution: Check CSV file path or scene naming

## Expected Behavior After Fix

When working correctly, you should see:
```
✓ Agent future trajectories found
  Shape: (20, 12)
  Mask shape: (20, 6)
  Non-zero mask values: 80/120  # ~67% of agents have future data
  Percentage: 66.67%
```

**Why not 100%?**
- Some agents may leave the scene before 6 future frames
- Some agents may be at scene boundaries
- 60-80% is typical for real-world scenarios

## Next Steps

1. Run validation script: `python validate_collision_detection.py`
2. Check debug output for specific error messages
3. Based on output, apply appropriate fix:
   - Adjust timestamp tolerance
   - Fix track_id format matching
   - Verify CSV loading

4. Re-run validation until masks are non-zero
5. Run full evaluation: collision rate should be 15-25% (not 0%)

## Files Modified

- `mmcv/datasets/suscape_orion_dataset.py`
  - Lines 1055-1082: Added debug logging and approximate timestamp matching
  - Lines 1085-1136: Added debug logging for track_id matching

## Testing

After getting non-zero masks, verify collision detection works:
```bash
python adzoo/orion/test.py \
    adzoo/orion/configs/suscape_eval.py \
    ckpts/orion_stage3.pth \
    --eval bbox
```

Expected metrics:
- Collision Rate: 15-25% (was 0%)
- L2 errors: Accurate trajectory metrics
