#!/usr/bin/env python3
"""
Verify the downloaded LiveCodeBench dataset and show its structure.
"""

import json
import os
from pathlib import Path
import glob


def verify_dataset():
    """Verify the downloaded dataset and show its structure."""
    dataset_path = Path("data/livecodebench_code_generation_lite")
    
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        return False
    
    print("🔍 Verifying LiveCodeBench dataset...")
    print(f"Dataset path: {dataset_path}")
    
    # Check data files
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    print(f"\n📁 Found {len(jsonl_files)} data files:")
    
    total_size = 0
    for file_path in jsonl_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {file_path.name}: {size_mb:.1f} MB")
    
    print(f"\n📊 Total dataset size: {total_size:.1f} MB")
    
    # Check dataset info
    info_path = Path("data/dataset_info.json")
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        print(f"\n📋 Dataset info:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
    
    # Sample data verification
    print(f"\n🔬 Sample data verification:")
    sample_file = jsonl_files[0] if jsonl_files else None
    if sample_file:
        with open(sample_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                try:
                    sample_data = json.loads(first_line)
                    print(f"  - Sample fields: {list(sample_data.keys())}")
                    print(f"  - Platform: {sample_data.get('platform', 'N/A')}")
                    print(f"  - Difficulty: {sample_data.get('difficulty', 'N/A')}")
                    print(f"  - Question ID: {sample_data.get('question_id', 'N/A')}")
                except json.JSONDecodeError:
                    print("  - Error parsing JSON data")
    
    print(f"\n✅ Dataset verification completed!")
    return True


if __name__ == "__main__":
    verify_dataset()
