#!/usr/bin/env python3
"""
Simple QA loading debug - doesn't require full dataset initialization
"""
import json
import os
from pathlib import Path

# Configuration
qa_root = '/lab/haoq_lab/cse12311753/sharegpt_dataset/'

print("=" * 80)
print("Simple QA Data Structure Check")
print("=" * 80)

print(f"\nQA root: {qa_root}")

# Load a few QA files directly
qa_tasks = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q8', 'q9']

for task in qa_tasks:
    filename = f'suscape_NQA_{task}.json'
    filepath = os.path.join(qa_root, filename)
    
    if not os.path.exists(filepath):
        print(f"\n{task}: File not found: {filepath}")
        continue
    
    print(f"\n{task}: Loading {filename}...")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"  Total entries: {len(data)}")
        
        # Show first few entries
        for i in range(min(3, len(data))):
            item = data[i]
            item_id = item.get('id', 'N/A')
            convs = item.get('conversations', [])
            image = item.get('image', 'N/A')
            
            print(f"\n  Entry {i}:")
            print(f"    ID: {item_id}")
            print(f"    Conversations: {len(convs)}")
            print(f"    Image: {image}")
            
            # Parse the ID format
            if '_' in item_id:
                parts = item_id.split('_')
                print(f"    ID parts: {parts}")
                if len(parts) >= 2:
                    scene_part = parts[0]  # e.g., 'scene-000000'
                    if len(parts) == 3:  # Format: scene-000000_7_q1
                        frame_part = parts[1]  # e.g., '7'
                        task_part = parts[2]   # e.g., 'q1'
                        print(f"    Parsed: scene='{scene_part}', frame={frame_part}, task={task_part}")
                    else:  # Other formats
                        print(f"    Format: {len(parts)} parts")
    
    except Exception as e:
        print(f"  ERROR loading: {e}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print("""
Based on the QA data structure, the ID format appears to be:
  scene-XXXXXX_FRAME_qN

Where:
  - scene-XXXXXX is the scene name (e.g., scene-000000)
  - FRAME is the frame number as an integer (e.g., 7, 9)
  - qN is the task ID (e.g., q1, q2)

To match this with dataset info, we need:
  - info['folder'] should be 'scene-XXXXXX' format
  - info['frame_idx'] should be an integer matching FRAME

If your CSV data has different scene names or frame indices,
the QA lookup will fail.
""")

print("\nNext step: Check what your CSV data contains")
print("Run this command to see your CSV structure:")
print("  head -20 /lab/haoq_lab/cse12311753/suscape_scenes/*.csv")
print("\nOr check the first data sample:")
print("  Check what 'folder' and 'frame_idx' values your dataset has")

print("\nDone!")
