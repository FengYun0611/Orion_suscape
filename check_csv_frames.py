#!/usr/bin/env python3
"""Check what frame indices are in the CSV file."""

import pandas as pd
import os

csv_root = '/lab/haoq_lab/cse12311753/suscape_scenes/'
csv_file = os.path.join(csv_root, '0.csv')

if os.path.exists(csv_file):
    print(f"Reading: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    
    if 'FRAME' in df.columns:
        frames = df['FRAME'].unique()
        print(f"\nUnique FRAME values: {len(frames)}")
        print(f"First 20 frames: {sorted(frames)[:20]}")
        print(f"Last 20 frames: {sorted(frames)[-20:]}")
        print(f"Frame range: {min(frames)} to {max(frames)}")
    else:
        print("\nNo 'FRAME' column found!")
        print(f"Available columns: {list(df.columns)}")
else:
    print(f"CSV file not found: {csv_file}")
    print(f"\nTrying to find CSV files in {csv_root}:")
    if os.path.exists(csv_root):
        csv_files = [f for f in os.listdir(csv_root) if f.endswith('.csv')]
        print(f"Found CSV files: {csv_files[:10]}")
    else:
        print(f"Directory not found: {csv_root}")
