#!/usr/bin/env python
"""Debug script to check why agent_fut_masks are all zero."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from mmcv.datasets import SUScapeOrionDataset

print("="*80)
print("DEBUG: Future Trajectory Extraction")
print("="*80)

# Initialize dataset
dataset = SUScapeOrionDataset(
    data_root='/lab/haoq_lab/cse12311753/suscape_scenes/',
    csv_root='/lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/',
    qa_root='/lab/haoq_lab/cse12311753/sharegpt_dataset/',
    ann_file='data/suscape_infos/suscape_infos_test.pkl',
    pipeline=[],
    test_mode=True
)

print(f"Dataset loaded: {len(dataset.data_infos)} samples")

# Get first sample
idx = 0
info = dataset.data_infos[idx]
ann_info = dataset.get_ann_info(idx)

print(f"\nSample {idx} info:")
print(f"  Scene: {info.get('folder', 'N/A')}")
print(f"  Timestamp: {info.get('timestamp', 'N/A')}")
print(f"  Num agents: {len(info.get('gt_ids', []))}")
print(f"  Agent IDs: {info.get('gt_ids', [])[:5]}")

print(f"\nAnnotation info:")
print(f"  agent_fut_traj shape: {ann_info['agent_fut_traj'].shape}")
print(f"  agent_fut_mask shape: {ann_info['agent_fut_mask'].shape}")
print(f"  Non-zero masks: {np.sum(ann_info['agent_fut_mask'])}/{ann_info['agent_fut_mask'].size}")

# Debug the extraction function directly
scene_name = info.get('folder', '')
current_timestamp = info.get('timestamp', None)
gt_ids = info.get('gt_ids', [])
gt_boxes = info.get('gt_boxes', np.array([]))
ego_x, ego_y = info.get('ego_translation', np.array([0.0, 0.0, 0.0]))[:2]
ego_yaw = info.get('ego_yaw', 0.0)

print(f"\nDirect extraction debug:")
print(f"  scene_name: {scene_name}")
print(f"  current_timestamp: {current_timestamp}")
print(f"  ego_x, ego_y, ego_yaw: {ego_x}, {ego_y}, {ego_yaw}")
print(f"  gt_ids: {gt_ids[:3]}")

# Check if scene CSV is loaded
if scene_name in dataset.scene_csv_data:
    df = dataset.scene_csv_data[scene_name]
    print(f"\n  CSV loaded for scene: {len(df)} rows")
    print(f"  CSV timestamps: {sorted(df['TIMESTAMP'].unique())[:10]}")
    print(f"  CSV track_ids (first 10): {df['TRACK_ID'].unique()[:10]}")
    
    # Check if current timestamp exists
    current_data = df[df['TIMESTAMP'] == current_timestamp]
    print(f"\n  Rows at current timestamp {current_timestamp}: {len(current_data)}")
    
    if len(gt_ids) > 0:
        test_track = gt_ids[0]
        print(f"\n  Testing track: {test_track}")
        
        # Check current timestamp
        track_current = df[(df['TIMESTAMP'] == current_timestamp) & (df['TRACK_ID'] == test_track)]
        print(f"    At current timestamp: {len(track_current)} rows")
        if len(track_current) > 0:
            print(f"      X={track_current.iloc[0]['X']}, Y={track_current.iloc[0]['Y']}")
        
        # Check future timestamps
        all_timestamps = sorted(df['TIMESTAMP'].unique())
        try:
            current_idx = all_timestamps.index(current_timestamp)
            future_timestamps = all_timestamps[current_idx + 1 : current_idx + 7]
            print(f"\n    Future timestamps: {future_timestamps}")
            
            for i, fut_ts in enumerate(future_timestamps):
                track_future = df[(df['TIMESTAMP'] == fut_ts) & (df['TRACK_ID'] == test_track)]
                print(f"      [{i}] ts={fut_ts}: {len(track_future)} rows", end='')
                if len(track_future) > 0:
                    print(f" - X={track_future.iloc[0]['X']:.2f}, Y={track_future.iloc[0]['Y']:.2f}")
                else:
                    print(" - NOT FOUND")
        except ValueError as e:
            print(f"    ERROR: Current timestamp not in list: {e}")
else:
    print(f"\n  ERROR: Scene {scene_name} not in scene_csv_data!")
    print(f"  Available scenes: {list(dataset.scene_csv_data.keys())[:5]}")

print("\n" + "="*80)
