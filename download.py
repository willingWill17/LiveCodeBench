#!/usr/bin/env python3
"""
Download script for LiveCodeBench dataset from Hugging Face.
Downloads the code_generation_lite dataset and saves it locally.
"""

import os
import json
from pathlib import Path
import argparse
import requests
import zipfile
import tempfile
import subprocess
import sys


def download_livecodebench_dataset(version_tag="release_v5", output_dir="data"):
    """
    Download LiveCodeBench dataset from Hugging Face.
    
    Args:
        version_tag (str): Version of the dataset to download
        output_dir (str): Directory to save the dataset
    """
    print(f"Downloading LiveCodeBench dataset (version: {version_tag})...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Method 1: Try using datasets library with trust_remote_code
    try:
        from datasets import load_dataset
        print("Method 1: Attempting to load dataset using datasets library...")
        
        # Try different approaches
        try:
            dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version_tag, trust_remote_code=True)
        except:
            try:
                dataset = load_dataset("livecodebench/code_generation_lite", trust_remote_code=True)
            except:
                dataset = load_dataset("livecodebench/code_generation_lite")
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset info: {dataset}")
        
        # Save the dataset locally
        dataset_path = output_path / "livecodebench_code_generation_lite"
        dataset.save_to_disk(str(dataset_path))
        
        print(f"Dataset saved to: {dataset_path}")
        
        # Save dataset info
        info_path = output_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump({
                "dataset_name": "livecodebench/code_generation_lite",
                "version_tag": version_tag,
                "download_method": "datasets_library",
                "status": "downloaded",
                "splits": list(dataset.keys()) if hasattr(dataset, 'keys') else [],
                "num_examples": {split: len(dataset[split]) for split in dataset.keys()} if hasattr(dataset, 'keys') else {}
            }, f, indent=2)
        
        print(f"Dataset info saved to: {info_path}")
        return True
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try using git clone
    try:
        print("\nMethod 2: Attempting to clone dataset repository...")
        dataset_path = output_path / "livecodebench_code_generation_lite"
        
        if dataset_path.exists():
            print(f"Removing existing directory: {dataset_path}")
            import shutil
            shutil.rmtree(dataset_path)
        
        # Clone the repository
        clone_cmd = ["git", "clone", "https://huggingface.co/datasets/livecodebench/code_generation_lite", str(dataset_path)]
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Dataset cloned successfully to: {dataset_path}")
            
            # Save dataset info
            info_path = output_path / "dataset_info.json"
            with open(info_path, 'w') as f:
                json.dump({
                    "dataset_name": "livecodebench/code_generation_lite",
                    "version_tag": version_tag,
                    "download_method": "git_clone",
                    "status": "downloaded"
                }, f, indent=2)
            
            print(f"Dataset info saved to: {info_path}")
            return True
        else:
            print(f"Git clone failed: {result.stderr}")
            
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Manual download instructions
    print("\nMethod 3: Providing manual download instructions...")
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("The automatic download methods failed. Please try the following:")
    print("\n1. Visit: https://huggingface.co/datasets/livecodebench/code_generation_lite")
    print("2. Click on 'Files and versions' tab")
    print("3. Download the dataset files manually")
    print("\nOr use the following commands:")
    print("\n# Option A: Using git")
    print("git clone https://huggingface.co/datasets/livecodebench/code_generation_lite data/livecodebench_code_generation_lite")
    print("\n# Option B: Using wget/curl")
    print("mkdir -p data/livecodebench_code_generation_lite")
    print("cd data/livecodebench_code_generation_lite")
    print("wget https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/dataset_info.json")
    print("wget https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/dataset_dict.json")
    print("wget https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/state.json")
    print("\n# Option C: Using Python requests")
    print("python -c \"import requests; import json; r = requests.get('https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/dataset_info.json'); print(json.dumps(r.json(), indent=2))\"")
    print("="*60)
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Download LiveCodeBench dataset")
    parser.add_argument(
        "--version", 
        default="release_v5", 
        help="Dataset version to download (default: release_v5)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data", 
        help="Output directory for the dataset (default: data)"
    )
    
    args = parser.parse_args()
    
    success = download_livecodebench_dataset(
        version_tag=args.version,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n✅ Dataset download completed successfully!")
    else:
        print("\n❌ Dataset download failed!")
        print("Please follow the manual instructions above.")
        exit(1)


if __name__ == "__main__":
    main()
