#!/usr/bin/env python3
"""
Debug script to check if QA data is loading correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mmcv.datasets import SUScapeOrionDataset
import json

# Create dataset instance
config_data_root = '/lab/haoq_lab/cse12311753/suscape_scenes/'
config_csv_root = '/lab/haoq_lab/cse12311753/suscape_scenes/'
config_qa_root = '/lab/haoq_lab/cse12311753/sharegpt_dataset/'

print("=" * 80)
print("QA Loading Debug Script")
print("=" * 80)

print(f"\nData root: {config_data_root}")
print(f"CSV root: {config_csv_root}")
print(f"QA root: {config_qa_root}")

# Initialize dataset with QA loading
dataset = SUScapeOrionDataset(
    data_root=config_data_root,
    csv_root=config_csv_root,
    qa_root=config_qa_root,
    ann_file='dummy.pkl',  # Will be generated
    pipeline=[],
    test_mode=True,
)

print("\n" + "=" * 80)
print("QA Data Structure Check")
print("=" * 80)

# Check what tasks were loaded
print(f"\nTasks loaded: {list(dataset.qa_data.keys())}")

# For each task, show how many scenes and frames
for task in sorted(dataset.qa_data.keys()):
    num_scenes = len(dataset.qa_data[task])
    total_frames = sum(len(frames) for frames in dataset.qa_data[task].values())
    print(f"\nTask {task}:")
    print(f"  Scenes: {num_scenes}")
    print(f"  Total frames: {total_frames}")
    
    # Show first few scene names
    scene_names = list(dataset.qa_data[task].keys())[:5]
    print(f"  Sample scene names: {scene_names}")
    
    # For first scene, show frame indices
    if scene_names:
        first_scene = scene_names[0]
        frame_indices = sorted(list(dataset.qa_data[task][first_scene].keys()))[:10]
        print(f"  Sample frame indices in '{first_scene}': {frame_indices}")

print("\n" + "=" * 80)
print("Dataset Info Check")
print("=" * 80)

# Check first few data_infos
print(f"\nTotal samples in dataset: {len(dataset.data_infos)}")

for i in range(min(5, len(dataset.data_infos))):
    info = dataset.data_infos[i]
    folder = info.get('folder', 'N/A')
    frame_idx = info.get('frame_idx', 'N/A')
    print(f"\nSample {i}:")
    print(f"  folder (scene_name): {folder}")
    print(f"  frame_idx: {frame_idx}")
    
    # Try to get QA for this sample
    qa_convs = dataset.get_qa_conversations(folder, frame_idx)
    print(f"  QA conversations found: {len(qa_convs)}")
    
    if len(qa_convs) == 0:
        # Debug why
        print(f"  Debug: Checking qa_data structure...")
        for task in ['q1', 'q2', 'q3']:
            if task in dataset.qa_data:
                has_scene = folder in dataset.qa_data[task]
                print(f"    {task}: scene '{folder}' exists = {has_scene}")
                if has_scene:
                    has_frame = frame_idx in dataset.qa_data[task][folder]
                    print(f"    {task}: frame {frame_idx} exists = {has_frame}")
                    available_frames = sorted(list(dataset.qa_data[task][folder].keys()))[:5]
                    print(f"    {task}: available frames (first 5): {available_frames}")

print("\n" + "=" * 80)
print("Manual Lookup Test")
print("=" * 80)

# Try manual lookup
test_scene = 'scene-000000'
test_frame = 7

print(f"\nTesting lookup for scene='{test_scene}', frame={test_frame}")

for task in ['q1', 'q2', 'q3']:
    if task in dataset.qa_data:
        if test_scene in dataset.qa_data[task]:
            if test_frame in dataset.qa_data[task][test_scene]:
                qa_item = dataset.qa_data[task][test_scene][test_frame]
                print(f"  {task}: FOUND")
                print(f"    ID: {qa_item.get('id', 'N/A')}")
                print(f"    Conversations: {len(qa_item.get('conversations', []))}")
            else:
                avail_frames = sorted(list(dataset.qa_data[task][test_scene].keys()))[:10]
                print(f"  {task}: Frame {test_frame} NOT FOUND in scene '{test_scene}'")
                print(f"    Available frames: {avail_frames}")
        else:
            avail_scenes = sorted(list(dataset.qa_data[task].keys()))[:5]
            print(f"  {task}: Scene '{test_scene}' NOT FOUND")
            print(f"    Available scenes: {avail_scenes}")

print("\nDone!")
