#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUScape Dataset Structure Verification Script

This script helps verify that your SUScape dataset is properly structured
for ORION open-loop evaluation.

Usage:
    python tools/verify_suscape_dataset.py data/suscape_scenes
"""

import os
import sys
import pandas as pd
from pathlib import Path


def check_csv_format(csv_path):
    """Check if CSV file has the correct format."""
    try:
        df = pd.read_csv(csv_path, sep='\t')
        required_columns = [
            'TIMESTAMP', 'TRACK_ID', 'OBJECT_TYPE', 'X', 'Y', 
            'V_X', 'V_Y', 'A_X', 'A_Y', 'YAW', 'DYAW', 'DDYAW', 'CITY_NAME'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        # Check if there's ego vehicle data
        if 'ego' not in df['TRACK_ID'].values:
            return False, "No ego vehicle data found (TRACK_ID='ego')"
        
        return True, f"Valid CSV with {len(df)} rows and {len(df['TIMESTAMP'].unique())} timestamps"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"


def check_camera_dirs(scene_dir):
    """Check if all required camera directories exist."""
    camera_names = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    
    results = {}
    for cam_name in camera_names:
        cam_dir = scene_dir / cam_name
        if cam_dir.exists():
            # Count image files
            img_files = list(cam_dir.glob('*.jpg')) + list(cam_dir.glob('*.png'))
            results[cam_name] = (True, len(img_files))
        else:
            results[cam_name] = (False, 0)
    
    return results


def verify_scene(scene_dir):
    """Verify a single scene directory."""
    print(f"\n  Checking scene: {scene_dir.name}")
    
    # Check CSV file
    csv_path = scene_dir / '0.csv'
    if not csv_path.exists():
        print(f"    ❌ CSV file not found: 0.csv")
        return False
    
    csv_ok, csv_msg = check_csv_format(csv_path)
    if csv_ok:
        print(f"    ✓ CSV format: {csv_msg}")
    else:
        print(f"    ❌ CSV format: {csv_msg}")
        return False
    
    # Check camera directories
    cam_results = check_camera_dirs(scene_dir)
    all_cams_ok = True
    for cam_name, (exists, num_imgs) in cam_results.items():
        if exists:
            print(f"    ✓ {cam_name}: {num_imgs} images")
        else:
            print(f"    ⚠ {cam_name}: Not found")
            all_cams_ok = False
    
    if not all_cams_ok:
        print(f"    ⚠ Warning: Some camera directories are missing")
    
    return csv_ok


def verify_suscape_dataset(data_root):
    """Verify the entire SUScape dataset."""
    data_root = Path(data_root)
    raws_dir = data_root / 'raws'
    
    print("=" * 60)
    print("SUScape Dataset Structure Verification")
    print("=" * 60)
    
    # Check if raws directory exists
    if not raws_dir.exists():
        print(f"\n❌ ERROR: 'raws' directory not found at {raws_dir}")
        print(f"   Expected structure:")
        print(f"   {data_root}/")
        print(f"   └── raws/")
        print(f"       ├── scene-000000/")
        print(f"       ├── scene-000001/")
        print(f"       └── ...")
        return False
    
    print(f"\n✓ Found raws directory: {raws_dir}")
    
    # Find all scene directories
    scene_dirs = sorted([d for d in raws_dir.iterdir() if d.is_dir() and d.name.startswith('scene-')])
    
    if not scene_dirs:
        print(f"\n❌ ERROR: No scene directories found in {raws_dir}")
        print(f"   Scene directories should be named 'scene-XXXXXX'")
        return False
    
    print(f"✓ Found {len(scene_dirs)} scene directories")
    
    # Verify each scene
    valid_scenes = 0
    for scene_dir in scene_dirs:
        if verify_scene(scene_dir):
            valid_scenes += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY")
    print("=" * 60)
    print(f"Total scenes: {len(scene_dirs)}")
    print(f"Valid scenes: {valid_scenes}")
    print(f"Invalid scenes: {len(scene_dirs) - valid_scenes}")
    
    if valid_scenes == len(scene_dirs):
        print("\n✓ All scenes are valid!")
        print("\nYou can now run:")
        print(f"  ./adzoo/orion/orion_dist_eval.sh adzoo/orion/configs/suscape_eval.py ckpts/Orion.pth 1")
    elif valid_scenes > 0:
        print(f"\n⚠ {len(scene_dirs) - valid_scenes} scenes have issues. Check the messages above.")
        print(f"  You can still proceed with {valid_scenes} valid scenes.")
    else:
        print(f"\n❌ No valid scenes found. Please fix the issues above.")
    
    return valid_scenes > 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_suscape_dataset.py <path_to_suscape_scenes>")
        print("Example: python verify_suscape_dataset.py data/suscape_scenes")
        sys.exit(1)
    
    data_root = sys.argv[1]
    success = verify_suscape_dataset(data_root)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
