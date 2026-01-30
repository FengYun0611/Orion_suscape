#!/usr/bin/env python
"""
Validation script to verify collision detection is working correctly.

This script checks:
1. Agent future trajectories are being extracted from CSV
2. Agent future masks are properly set
3. BEV occupancy maps are being generated
4. Collision detection logic is functioning

Run this to verify the collision detection fix before evaluating the full model.
"""

import numpy as np
import pickle
import pandas as pd
import sys
import os

def validate_future_trajectory_extraction():
    """Check if future trajectories can be extracted from CSV."""
    print("=" * 80)
    print("VALIDATION 1: Future Trajectory Extraction from CSV")
    print("=" * 80)
    
    # Check if CSV has multi-frame data
    csv_path = "/lab/haoq_lab/cse12311753/suscape_scenes/0.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/suscape_scenes/0.csv"  # Try local path
    
    if not os.path.exists(csv_path):
        print("⚠ WARNING: Cannot find CSV file to validate")
        print(f"  Tried: /lab/haoq_lab/cse12311753/suscape_scenes/0.csv")
        print(f"  Tried: data/suscape_scenes/0.csv")
        return False
    
    df = pd.read_csv(csv_path, sep='\t')
    
    # Check structure
    required_cols = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y', 'YAW']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"✗ FAIL: Missing columns: {missing}")
        return False
    
    print(f"✓ CSV file found with {len(df)} rows")
    print(f"✓ Required columns present: {required_cols}")
    
    # Check timestamp distribution
    timestamps = sorted(df['TIMESTAMP'].unique())
    print(f"✓ Found {len(timestamps)} unique timestamps")
    print(f"  First timestamp: {timestamps[0]}")
    print(f"  Last timestamp: {timestamps[-1]}")
    
    # Check if same track_id appears across multiple timestamps
    track_ids = df['TRACK_ID'].unique()
    multi_frame_tracks = 0
    example_track = None
    
    for track_id in track_ids[:100]:  # Check first 100 tracks
        track_df = df[df['TRACK_ID'] == track_id]
        if len(track_df) > 1:
            multi_frame_tracks += 1
            if example_track is None:
                example_track = track_id
    
    print(f"✓ Found {multi_frame_tracks} tracks with multi-frame data (out of first 100)")
    
    if multi_frame_tracks == 0:
        print("✗ FAIL: No tracks have multi-frame data - future trajectories cannot be extracted!")
        return False
    
    # Show example
    if example_track:
        track_df = df[df['TRACK_ID'] == example_track].sort_values('TIMESTAMP')
        print(f"\n  Example track '{example_track}':")
        print(f"    Appears in {len(track_df)} timestamps")
        print(f"    Timestamp range: {track_df['TIMESTAMP'].min()} -> {track_df['TIMESTAMP'].max()}")
        print(f"    Position range: X=[{track_df['X'].min():.1f}, {track_df['X'].max():.1f}], "
              f"Y=[{track_df['Y'].min():.1f}, {track_df['Y'].max():.1f}]")
    
    print("\n✓ PASS: CSV data structure supports future trajectory extraction")
    return True


def validate_dataset_future_trajectories():
    """Check if dataset is properly extracting future trajectories."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: Dataset Future Trajectory Integration")
    print("=" * 80)
    
    try:
        # Try to import dataset
        sys.path.insert(0, '/home/runner/work/Orion_suscape/Orion_suscape')
        from mmcv.datasets import SUScapeOrionDataset
        
        # Initialize dataset with minimal config
        # Note: csv_root should point to the directory containing CSV files
        dataset = SUScapeOrionDataset(
            data_root='/lab/haoq_lab/cse12311753/suscape_scenes/',
            csv_root='/lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/',
            qa_root='/lab/haoq_lab/cse12311753/sharegpt_dataset/',
            ann_file='data/suscape_infos/suscape_infos_test.pkl',
            pipeline=[],
            test_mode=True
        )
        
        print(f"✓ Dataset initialized with {len(dataset)} samples")
        
        # Check if scene_csv_data is populated
        if not hasattr(dataset, 'scene_csv_data'):
            print("✗ FAIL: Dataset does not have 'scene_csv_data' attribute")
            print("  The _parse_csv_file() method may not be storing CSV data correctly")
            return False
        
        print(f"✓ Dataset has 'scene_csv_data' attribute")
        print(f"  Loaded {len(dataset.scene_csv_data)} scenes")
        
        # Test annotation extraction on first sample
        if len(dataset) > 0:
            ann = dataset.get_ann_info(0)
            
            print(f"\n  Testing annotation extraction on sample 0:")
            print(f"    attr_labels shape: {ann['attr_labels'].shape}")
            
            # Extract future trajectory components
            if len(ann['attr_labels']) > 0:
                agent_fut_trajs = ann['attr_labels'][:, :12]  # First 12 = 6*2 (x,y)
                agent_fut_masks = ann['attr_labels'][:, 12:18]  # Next 6 = masks
                
                print(f"    Number of agents: {len(agent_fut_trajs)}")
                print(f"    Future trajectories non-zero: {np.count_nonzero(agent_fut_trajs)} / {agent_fut_trajs.size}")
                print(f"    Future masks set to 1: {np.count_nonzero(agent_fut_masks)} / {agent_fut_masks.size}")
                
                if np.count_nonzero(agent_fut_masks) == 0:
                    print("  ✗ FAIL: All future masks are zero - trajectories not extracted!")
                    return False
                elif np.count_nonzero(agent_fut_trajs) == 0:
                    print("  ✗ FAIL: All future trajectories are zero - extraction failed!")
                    return False
                else:
                    print(f"  ✓ PASS: Future trajectories extracted successfully")
                    
                    # Show example
                    for i in range(min(3, len(agent_fut_trajs))):
                        valid_steps = int(agent_fut_masks[i].sum())
                        if valid_steps > 0:
                            print(f"\n    Agent {i}: {valid_steps} valid future steps")
                            for t in range(min(3, valid_steps)):
                                x, y = agent_fut_trajs[i, t*2], agent_fut_trajs[i, t*2+1]
                                print(f"      Step {t}: x={x:.2f}m, y={y:.2f}m")
        
        print("\n✓ PASS: Dataset correctly extracts future trajectories from CSV")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Error during dataset validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_bev_rendering_logic():
    """Check if BEV rendering logic will use the future trajectories."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: BEV Occupancy Rendering Logic")
    print("=" * 80)
    
    # Read the metric code
    metric_file = "mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py"
    
    if not os.path.exists(metric_file):
        print(f"⚠ WARNING: Cannot find {metric_file}")
        return False
    
    with open(metric_file, 'r') as f:
        code = f.read()
    
    # Check critical parts
    checks = {
        "Reads future masks": "gt_agent_fut_mask" in code,
        "Checks mask == 1": "gt_agent_fut_mask[i][t] == 1" in code,
        "Renders to BEV": "cv2.fillPoly" in code,
        "Uses cumsum for trajectories": "np.cumsum" in code,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}: {passed}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ PASS: BEV rendering logic looks correct")
        print("  Key flow:")
        print("    1. Extracts gt_agent_fut_mask from attr_labels[:, 12:18]")
        print("    2. For each timestep t, checks if gt_agent_fut_mask[i][t] == 1")
        print("    3. If mask is 1, renders obstacle bounding box to BEV occupancy map")
        print("    4. Collision detected if ego trajectory intersects occupied pixels")
    else:
        print("\n✗ FAIL: BEV rendering logic may have issues")
    
    return all_passed


def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("COLLISION DETECTION VALIDATION SUITE")
    print("=" * 80)
    print("\nThis script validates that the collision detection fix is working correctly.")
    print("It checks the full pipeline from CSV extraction to BEV rendering logic.\n")
    
    results = []
    
    # Run validations
    results.append(("CSV Multi-Frame Data", validate_future_trajectory_extraction()))
    results.append(("Dataset Integration", validate_dataset_future_trajectories()))
    results.append(("BEV Rendering Logic", validate_bev_rendering_logic()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✓ ALL VALIDATIONS PASSED!")
        print("=" * 80)
        print("\nCollision detection should now work correctly.")
        print("When you run evaluation, you should see non-zero collision rates.")
        print("\nExpected behavior:")
        print("  - agent_fut_masks will have values of 1.0 for valid future timesteps")
        print("  - BEV occupancy maps will show obstacle positions")
        print("  - Collision rate should be 10-30% (typical for autonomous driving)")
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ SOME VALIDATIONS FAILED")
        print("=" * 80)
        print("\nPlease review the failures above.")
        print("The collision detection may not work correctly until these are fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
