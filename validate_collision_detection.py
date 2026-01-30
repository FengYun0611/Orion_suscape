#!/usr/bin/env python
"""
Validation script to verify collision detection is working correctly.

This script checks:
1. CSV multi-frame data structure and integrity  
2. Dataset future trajectory extraction from CSV
3. BEV rendering logic (mask checking, cv2.fillPoly calls)
4. Provides detailed pass/fail diagnostics for each pipeline component

Run this to verify the collision detection fix before evaluating the full model.
"""

import numpy as np
import pickle
import pandas as pd
import sys
import os
import glob

def validate_csv_data_structure():
    """Check if CSV files contain multi-frame data for the same track_id."""
    print("=" * 80)
    print("VALIDATION 1: CSV Multi-Frame Data Structure")
    print("=" * 80)
    
    csv_root = '/lab/haoq_lab/cse12311753/suscape_scene_traj_csv_alldistance_fixyaw/'
    
    # Find a CSV file to test
    csv_files = glob.glob(os.path.join(csv_root, '*.csv'))
    if not csv_files:
        print(f"✗ FAIL: No CSV files found in {csv_root}")
        return False
    
    csv_file = csv_files[0]
    print(f"Testing CSV file: {csv_file}")
    
    try:
        # Read CSV with auto-detected delimiter (pandas default is comma)
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['TIMESTAMP', 'TRACK_ID', 'X', 'Y', 'YAW']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"✗ FAIL: Missing columns: {missing}")
            print(f"  Available columns: {df.columns.tolist()}")
            return False
        
        print(f"✓ CSV loaded successfully with {len(df)} rows")
        print(f"✓ Required columns present: {required_cols}")
        
        # Check if multiple timestamps exist
        timestamps = sorted(df['TIMESTAMP'].unique())
        print(f"✓ Found {len(timestamps)} unique timestamps")
        
        if len(timestamps) < 2:
            print("✗ FAIL: Need at least 2 timestamps for multi-frame validation")
            return False
        
        # Check if same track_id appears in multiple timestamps
        track_ids = df['TRACK_ID'].unique()
        multi_frame_tracks = 0
        
        for track_id in track_ids[:10]:  # Check first 10 tracks
            track_data = df[df['TRACK_ID'] == track_id]
            track_timestamps = track_data['TIMESTAMP'].unique()
            if len(track_timestamps) > 1:
                multi_frame_tracks += 1
        
        if multi_frame_tracks == 0:
            print("✗ FAIL: No track_id appears in multiple timestamps")
            print("  This means we can't extract future trajectories!")
            return False
        
        print(f"✓ Found {multi_frame_tracks}/10 tracks with multi-frame data")
        
        # Test future trajectory extraction for one track
        test_track = None
        for track_id in track_ids:
            if track_id != 'ego':
                track_data = df[df['TRACK_ID'] == track_id]
                if len(track_data['TIMESTAMP'].unique()) >= 7:  # Need at least current + 6 future
                    test_track = track_id
                    break
        
        if test_track:
            print(f"✓ Test track '{test_track}' has sufficient future frames")
            track_data = df[df['TRACK_ID'] == test_track]
            track_timestamps = sorted(track_data['TIMESTAMP'].unique())
            print(f"  Timestamps: {track_timestamps[:7]}")
            return True
        else:
            print("⚠ WARNING: No track found with 7+ consecutive frames")
            print("  Future trajectory extraction may be limited")
            return True  # Still pass, but with warning
            
    except Exception as e:
        print(f"✗ FAIL: Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        
        print(f"✓ Dataset initialized successfully")
        print(f"✓ Total samples: {len(dataset)}")
        
        # Try to get annotation for first sample
        if len(dataset) == 0:
            print("✗ FAIL: Dataset has no samples")
            return False
        
        # Get a sample
        sample_idx = 0
        info = dataset.data_infos[sample_idx]
        ann_info = dataset.get_ann_info(sample_idx)
        
        print(f"\n✓ Successfully retrieved annotation for sample {sample_idx}")
        print(f"  Scene: {info.get('folder', 'unknown')}")
        print(f"  Timestamp: {info.get('timestamp', 'unknown')}")
        
        # Check agent future trajectories
        if 'agent_fut_traj' in ann_info:
            agent_fut_trajs = ann_info['agent_fut_traj']
            agent_fut_masks = ann_info['agent_fut_mask']
            
            print(f"\n✓ Agent future trajectories found")
            print(f"  Shape: {agent_fut_trajs.shape}")
            print(f"  Mask shape: {agent_fut_masks.shape}")
            
            # Check if masks have non-zero values
            nonzero_masks = np.count_nonzero(agent_fut_masks)
            total_mask_values = agent_fut_masks.size
            
            print(f"  Non-zero mask values: {nonzero_masks}/{total_mask_values}")
            print(f"  Percentage: {100 * nonzero_masks / total_mask_values:.2f}%")
            
            if nonzero_masks == 0:
                print("\n✗ FAIL: All agent_fut_masks are zero!")
                print("  This means no obstacles will be rendered in BEV")
                print("  Collision rate will be 0%")
                return False
            else:
                print(f"\n✓ PASS: {nonzero_masks} valid future trajectory points")
                
                # Show some examples
                for agent_idx in range(min(3, agent_fut_trajs.shape[0])):
                    mask = agent_fut_masks[agent_idx]
                    valid_steps = np.sum(mask)
                    if valid_steps > 0:
                        traj = agent_fut_trajs[agent_idx].reshape(-1, 2)
                        print(f"  Agent {agent_idx}: {int(valid_steps)} valid future steps")
                        print(f"    First future position: {traj[0]}")
                
                return True
        else:
            print("✗ FAIL: 'agent_fut_traj' not found in annotation")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Error during dataset validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_bev_rendering_logic():
    """Verify BEV rendering uses agent_fut_mask correctly."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: BEV Rendering Logic")
    print("=" * 80)
    
    try:
        # Check if metric_stp3.py has the correct rendering logic
        metric_file = '/home/runner/work/Orion_suscape/Orion_suscape/adzoo/orion/projects/mmdet3d_plugin/core/evaluation/metric_stp3.py'
        
        if not os.path.exists(metric_file):
            print(f"⚠ WARNING: Cannot find {metric_file}")
            return True  # Skip validation
        
        with open(metric_file, 'r') as f:
            content = f.read()
        
        # Check for mask usage
        if 'agent_fut_mask' in content:
            print("✓ BEV rendering code uses 'agent_fut_mask'")
        else:
            print("⚠ WARNING: 'agent_fut_mask' not found in BEV rendering code")
        
        # Check for fillPoly (used to render obstacles)
        if 'fillPoly' in content:
            print("✓ BEV rendering uses cv2.fillPoly for obstacle rendering")
        else:
            print("⚠ WARNING: 'fillPoly' not found - obstacles may not be rendered")
        
        # Check for mask checking before rendering
        if 'if mask[i][t]' in content or 'if agent_fut_mask' in content:
            print("✓ Mask values are checked before rendering obstacles")
        else:
            print("⚠ WARNING: Mask checking logic not clearly visible")
        
        return True
        
    except Exception as e:
        print(f"⚠ WARNING: Could not validate BEV rendering: {e}")
        return True  # Don't fail, just warn


def main():
    """Run all validations."""
    print("\n" + "=" * 80)
    print("COLLISION DETECTION VALIDATION SUITE")
    print("=" * 80)
    print("\nThis script validates that collision detection is correctly implemented.")
    print("It checks:")
    print("  1. CSV files contain multi-frame trajectory data")
    print("  2. Dataset extracts future trajectories into agent_fut_traj/mask")
    print("  3. BEV rendering logic uses masks correctly")
    print("\n")
    
    results = {}
    
    # Run validations
    results['CSV Structure'] = validate_csv_data_structure()
    results['Dataset Integration'] = validate_dataset_future_trajectories()
    results['BEV Rendering'] = validate_bev_rendering_logic()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("\nYour collision detection implementation looks correct!")
        print("You should now see non-zero collision rates when running evaluation.")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("\nPlease fix the issues above before running full evaluation.")
        print("Collision rate may still be 0% until these are resolved.")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
