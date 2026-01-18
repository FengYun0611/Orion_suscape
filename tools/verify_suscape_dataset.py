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


def verify_suscape_dataset(data_root, csv_root=None):
    """Verify the entire SUScape dataset.
    
    Args:
        data_root: Path to directory containing scene folders
        csv_root: Optional path to directory containing CSV files (for new structure)
    """
    data_root = Path(data_root)
    
    print("=" * 60)
    print("SUScape Dataset Structure Verification")
    print("=" * 60)
    
    # Check for old structure (with raws/ subdirectory)
    raws_dir = data_root / 'raws'
    if raws_dir.exists():
        print(f"\n✓ Found 'raws' directory - using OLD structure")
        print(f"  Scenes location: {raws_dir}")
        print(f"  CSV location: Inside each scene directory (0.csv)")
        scene_parent = raws_dir
        structure_type = 'old'
    else:
        # Check for new structure (scenes directly in data_root)
        scene_dirs_test = sorted([d for d in data_root.iterdir() 
                                 if d.is_dir() and d.name.startswith('scene-')])
        if scene_dirs_test:
            print(f"\n✓ Found scenes directly in data_root - using NEW structure")
            print(f"  Scenes location: {data_root}")
            if csv_root:
                csv_path = Path(csv_root)
                if csv_path.exists():
                    print(f"  CSV location: {csv_root}")
                else:
                    print(f"\n❌ ERROR: Specified csv_root not found: {csv_root}")
                    return False
            else:
                print(f"\n⚠ WARNING: csv_root not specified for new structure")
                print(f"  Please provide csv_root parameter pointing to CSV directory")
                return False
            scene_parent = data_root
            structure_type = 'new'
        else:
            print(f"\n❌ ERROR: No scene directories found")
            print(f"   Checked for:")
            print(f"   - Old structure: {raws_dir}/scene-XXXXXX/")
            print(f"   - New structure: {data_root}/scene-XXXXXX/")
            return False
    
    # Find all scene directories
    scene_dirs = sorted([d for d in scene_parent.iterdir() 
                        if d.is_dir() and d.name.startswith('scene-')])
    
    if not scene_dirs:
        print(f"\n❌ ERROR: No scene directories found in {scene_parent}")
        return False
    
    print(f"✓ Found {len(scene_dirs)} scene directories")
    
    # Verify each scene based on structure type
    valid_scenes = 0
    if structure_type == 'old':
        for scene_dir in scene_dirs:
            if verify_scene(scene_dir):
                valid_scenes += 1
    else:  # new structure
        csv_path = Path(csv_root)
        for scene_dir in scene_dirs:
            # Extract scene number
            try:
                scene_num = int(scene_dir.name.split('-')[1])
                csv_file = csv_path / f'{scene_num}.csv'
                
                print(f"\n  Checking scene: {scene_dir.name}")
                
                # Check CSV
                if not csv_file.exists():
                    print(f"    ❌ CSV file not found: {csv_file}")
                    continue
                
                csv_ok, csv_msg = check_csv_format(csv_file)
                if csv_ok:
                    print(f"    ✓ CSV format: {csv_msg}")
                else:
                    print(f"    ❌ CSV format: {csv_msg}")
                    continue
                
                # Check cameras
                cam_results = check_camera_dirs(scene_dir)
                all_cams_ok = True
                for cam_name, (exists, num_imgs) in cam_results.items():
                    if exists:
                        print(f"    ✓ {cam_name}: {num_imgs} images")
                    else:
                        print(f"    ⚠ {cam_name}: Not found")
                        all_cams_ok = False
                
                if csv_ok:
                    valid_scenes += 1
                    
            except (IndexError, ValueError):
                print(f"  ❌ Cannot parse scene number from {scene_dir.name}")
                continue
    
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
        print("Usage: python verify_suscape_dataset.py <path_to_scenes> [csv_root]")
        print("Examples:")
        print("  Old structure: python verify_suscape_dataset.py data/suscape_scenes")
        print("  New structure: python verify_suscape_dataset.py data/suscape_scenes data/suscape_scene_traj_csv_alldistance_fixyaw")
        sys.exit(1)
    
    data_root = sys.argv[1]
    csv_root = sys.argv[2] if len(sys.argv) > 2 else None
    success = verify_suscape_dataset(data_root, csv_root)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
