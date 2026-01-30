#!/usr/bin/env python3
"""
Final comprehensive QA debugging script.
This will show exactly why QA conversations are not loading.
"""

import json
import os
import os.path as osp

print("="*80)
print("Final QA Debugging Script")
print("="*80)
print()

# Paths
qa_root = '/lab/haoq_lab/cse12311753/sharegpt_dataset/'
csv_file = '/lab/haoq_lab/cse12311753/suscape_scenes/0.csv'

# 1. Load QA data and show indexing
print("STEP 1: Loading QA data (q1 as example)")
print("-" * 80)

qa_file = osp.join(qa_root, 'suscape_NQA_q1.json')
with open(qa_file, 'r') as f:
    qa_list = json.load(f)

print(f"Total QA entries: {len(qa_list)}")
print()

# Build index like the code does
qa_index_by_timestamp = {}
qa_index_by_frame = {}

for qa_item in qa_list[:10]:  # First 10 for debug
    item_id = qa_item.get('id', '')
    image_path = qa_item.get('image', '')
    
    # Extract scene name
    parts = item_id.split('_')
    scene_name = parts[0] if parts else None
    frame_idx = int(parts[1]) if len(parts) >= 2 else None
    
    # Extract timestamp from image path
    filename = osp.basename(image_path)
    timestamp = None
    if filename.endswith('.jpg') or filename.endswith('.png'):
        try:
            timestamp_str = filename.rsplit('.', 2)[0]
            timestamp = int(float(timestamp_str))
        except (ValueError, IndexError):
            pass
    
    print(f"  ID: {item_id}")
    print(f"    Scene: {scene_name}, Frame: {frame_idx}")
    print(f"    Image: {image_path}")
    print(f"    Extracted timestamp: {timestamp}")
    
    if scene_name:
        if scene_name not in qa_index_by_timestamp:
            qa_index_by_timestamp[scene_name] = {}
            qa_index_by_frame[scene_name] = {}
        if timestamp:
            qa_index_by_timestamp[scene_name][timestamp] = item_id
        if frame_idx:
            qa_index_by_frame[scene_name][frame_idx] = item_id
    print()

print("\nStep 1 Summary:")
print(f"  Scenes found: {list(qa_index_by_timestamp.keys())}")
for scene in qa_index_by_timestamp:
    print(f"  {scene}:")
    print(f"    Timestamps indexed: {sorted(qa_index_by_timestamp[scene].keys())}")
    print(f"    Frame indices indexed: {sorted(qa_index_by_frame[scene].keys())}")
print()

# 2. Load CSV data and show what timestamps it has
print("STEP 2: Loading CSV data")
print("-" * 80)

import pandas as pd
df = pd.read_csv(csv_file, sep='\t')

# Get ego rows
ego_df = df[df['TRACK_ID'] == 'ego'].head(10)

print(f"Total rows in CSV: {len(df)}")
print(f"Ego rows (first 10): {len(ego_df)}")
print()

print("First 10 ego timestamps from CSV:")
for idx, row in ego_df.iterrows():
    print(f"  Timestamp: {row['TIMESTAMP']}")

print()

# 3. Check if any match
print("STEP 3: Checking for matches")
print("-" * 80)

csv_timestamps = set(ego_df['TIMESTAMP'].values)
scene_name = 'scene-000000'

if scene_name in qa_index_by_timestamp:
    qa_timestamps = set(qa_index_by_timestamp[scene_name].keys())
    
    print(f"CSV timestamps (first 10): {sorted(list(csv_timestamps))[:10]}")
    print(f"QA timestamps (first 10): {sorted(list(qa_timestamps))[:10]}")
    print()
    
    matches = csv_timestamps & qa_timestamps
    print(f"Number of matching timestamps: {len(matches)}")
    if matches:
        print(f"Matching timestamps: {sorted(list(matches))[:10]}")
    else:
        print("NO MATCHES FOUND!")
        print()
        print("This is the problem! The timestamps don't match.")
        print()
        print("Possible reasons:")
        print("1. QA data uses different timestamps than CSV")
        print("2. Scene name mismatch")
        print("3. Timestamp extraction logic is wrong")
        print()
        print("Let's compare the values:")
        print(f"  CSV first timestamp: {sorted(csv_timestamps)[0]}")
        print(f"  QA first timestamp: {sorted(qa_timestamps)[0]}")
        print(f"  Difference: {sorted(qa_timestamps)[0] - sorted(csv_timestamps)[0]}")
else:
    print(f"Scene {scene_name} not found in QA index!")
    print(f"Available scenes: {list(qa_index_by_timestamp.keys())}")

print()
print("="*80)
print("Debug complete. Use this information to fix the timestamp matching.")
print("="*80)
