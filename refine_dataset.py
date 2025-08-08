#!/usr/bin/env python3
"""
Refine LiveCodeBench dataset by extracting only specific fields.
This script processes all JSONL files and retains:
- question_id (for evaluation)
- question_title (short title for identification)
- question_content 
- difficulty (problem difficulty level)
- starter_code (important for LLM solving approach)
- public_test_cases
"""

import json
import os
from pathlib import Path
from datetime import datetime
import glob

def refine_dataset(input_file, output_file):
    """
    Refine the dataset by keeping only specified fields.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    print(f"🔍 Refining dataset...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    starter_code_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the original problem
                problem = json.loads(line)
                
                # Extract only the required fields
                refined_problem = {
                    'question_id': problem.get('question_id', ''),
                    'question_title': problem.get('question_title', ''),
                    'question_content': problem.get('question_content', ''),
                    'difficulty': problem.get('difficulty', ''),
                    'starter_code': problem.get('starter_code', ''),
                    'public_test_cases': problem.get('public_test_cases', '')
                }
                
                # Count problems with starter code
                if refined_problem['starter_code'].strip():
                    starter_code_count += 1
                
                # Write the refined problem to output file
                outfile.write(json.dumps(refined_problem, ensure_ascii=False) + '\n')
                processed_count += 1
                
                # Progress indicator
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} problems...")
                    
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Error parsing line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"  ❌ Unexpected error on line {line_num}: {e}")
                error_count += 1
                continue
    
    # Calculate file sizes
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Dataset refinement completed!")
    print(f"📊 Statistics:")
    print(f"  - Input file size: {input_size_mb:.1f} MB")
    print(f"  - Output file size: {output_size_mb:.1f} MB")
    print(f"  - Size reduction: {((input_size_mb - output_size_mb) / input_size_mb * 100):.1f}%")
    print(f"  - Problems processed: {processed_count}")
    print(f"  - Problems with starter code: {starter_code_count} ({starter_code_count/processed_count*100:.1f}%)")
    print(f"  - Errors encountered: {error_count}")
    
    # Verify the refined dataset
    print(f"\n🔍 Verifying refined dataset...")
    verify_refined_dataset(output_path)
    
    return True

def verify_refined_dataset(file_path):
    """Verify the refined dataset structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read first few lines to verify structure
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 3:  # Check first 3 lines
                    break
                sample_lines.append(json.loads(line.strip()))
            
            print(f"  ✅ Successfully parsed {len(sample_lines)} sample problems")
            
            # Check field structure
            if sample_lines:
                sample = sample_lines[0]
                expected_fields = {'question_id', 'question_title', 'question_content', 'difficulty', 'starter_code', 'public_test_cases'}
                actual_fields = set(sample.keys())
                
                if actual_fields == expected_fields:
                    print(f"  ✅ All expected fields present: {expected_fields}")
                else:
                    missing = expected_fields - actual_fields
                    extra = actual_fields - expected_fields
                    if missing:
                        print(f"  ⚠️  Missing fields: {missing}")
                    if extra:
                        print(f"  ⚠️  Extra fields: {extra}")
                
                # Show sample data
                print(f"  📋 Sample problem structure:")
                for field in expected_fields:
                    value = sample.get(field, '')
                    if field == 'public_test_cases':
                        try:
                            test_cases = json.loads(value) if isinstance(value, str) else value
                            print(f"    - {field}: {len(test_cases)} test cases")
                        except:
                            print(f"    - {field}: [Error parsing test cases]")
                    elif field == 'starter_code':
                        has_starter = "Yes" if value.strip() else "No"
                        print(f"    - {field}: {has_starter}")
                    else:
                        preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"    - {field}: {preview}")
                        
    except Exception as e:
        print(f"  ❌ Error verifying refined dataset: {e}")

def process_all_datasets():
    """Process all JSONL files in the dataset directory."""
    dataset_dir = Path("data/livecodebench_code_generation_lite")
    output_dir = Path("data/refined_data")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    # Find all JSONL files
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ No JSONL files found in {dataset_dir}")
        return False
    
    print(f"🔧 LiveCodeBench Dataset Refinement Tool")
    print("=" * 60)
    print("This script will refine all JSONL files to keep only:")
    print("  - question_id (for evaluation)")
    print("  - question_title (short title for identification)")
    print("  - question_content")
    print("  - difficulty (problem difficulty level)")
    print("  - starter_code (important for LLM solving approach)")
    print("  - public_test_cases")
    print("=" * 60)
    print(f"Found {len(jsonl_files)} files to process:")
    for file in jsonl_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.1f} MB")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_errors = 0
    
    for input_file in jsonl_files:
        output_file = output_dir / f"refined_{input_file.name}"
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_file.name}")
        print(f"{'='*60}")
        
        success = refine_dataset(str(input_file), str(output_file))
        if success:
            total_processed += 1
        else:
            total_errors += 1
    
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETED!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {total_processed} files")
    print(f"❌ Failed: {total_errors} files")
    print(f"📁 Output directory: {output_dir}")
    
    # Show final output files
    refined_files = list(output_dir.glob("refined_*.jsonl"))
    if refined_files:
        print(f"\n📋 Refined files created:")
        total_size = 0
        for file in refined_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {file.name}: {size_mb:.1f} MB")
        print(f"  Total size: {total_size:.1f} MB")
    
    return total_errors == 0

def main():
    """Main function to run the dataset refinement."""
    return process_all_datasets()

if __name__ == "__main__":
    main()
