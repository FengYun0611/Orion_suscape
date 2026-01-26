#!/usr/bin/env python3
"""
Split SUScape dataset pickle file into train/val splits for fine-tuning.

Usage:
    python tools/split_suscape_data.py \
        --input /lab/haoq_lab/cse12311753/suscape_scenes/test.pkl \
        --output-dir /lab/haoq_lab/cse12311753/suscape_scenes \
        --train-ratio 0.8

This will create:
    - train.pkl (80% of data)
    - val.pkl (20% of data)
"""

import argparse
import pickle
import random
from pathlib import Path


def split_pickle(input_file, output_dir, train_ratio=0.8, seed=42):
    """Split a SUScape pickle file into train and val sets."""
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get data_infos list
    if 'infos' in data:
        all_infos = data['infos']
        metadata = data.get('metadata', {})
    else:
        all_infos = data
        metadata = {}
    
    total_samples = len(all_infos)
    print(f"Total samples: {total_samples}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # Split indices
    split_point = int(total_samples * train_ratio)
    train_indices = set(indices[:split_point])
    val_indices = set(indices[split_point:])
    
    # Create train and val data
    train_infos = [all_infos[i] for i in range(total_samples) if i in train_indices]
    val_infos = [all_infos[i] for i in range(total_samples) if i in val_indices]
    
    train_data = {'infos': train_infos, 'metadata': metadata}
    val_data = {'infos': val_infos, 'metadata': metadata}
    
    # Save train.pkl
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / 'train.pkl'
    val_file = output_dir / 'val.pkl'
    
    print(f"\nSaving train set ({len(train_infos)} samples) to {train_file}...")
    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)
    
    print(f"Saving val set ({len(val_infos)} samples) to {val_file}...")
    with open(val_file, 'wb') as f:
        pickle.dump(val_data, f)
    
    print("\n=== Split Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_infos)} ({100*train_ratio:.1f}%)")
    print(f"Val samples: {len(val_infos)} ({100*(1-train_ratio):.1f}%)")
    print(f"Seed: {seed}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Split SUScape pickle data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file (e.g., test.pkl)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for train.pkl and val.pkl')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train set ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_pickle(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
