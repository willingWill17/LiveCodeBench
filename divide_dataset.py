#!/usr/bin/env python3
"""
Script to divide the refined dataset into individual question files.
Creates a structure like:
individual_questions/
├── test/
│   ├── question_id_1.txt
│   ├── question_id_2.txt
│   └── ...
├── test2/
│   ├── question_id_3.txt
│   └── ...
└── ...
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def create_individual_questions(input_dir: str, output_dir: str):
    """
    Divide the refined dataset into individual question files.
    
    Args:
        input_dir: Directory containing refined JSONL files
        output_dir: Directory to create individual question files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Process each JSONL file
    jsonl_files = list(input_path.glob("*.jsonl"))
    total_questions = 0
    
    print(f"Found {len(jsonl_files)} JSONL files to process...")
    
    for jsonl_file in jsonl_files:
        # Extract test folder name from filename (e.g., "refined_test.jsonl" -> "test")
        test_name = jsonl_file.stem.replace("refined_", "")
        test_dir = output_path / test_name
        
        # Create test directory
        test_dir.mkdir(exist_ok=True)
        
        questions_in_file = 0
        
        print(f"\nProcessing {jsonl_file.name} -> {test_name}/")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    question_data = json.loads(line.strip())
                    question_id = question_data.get('question_id', f'unknown_{line_num}')
                    
                    # Create individual question file
                    question_file = test_dir / f"{question_id}.txt"
                    
                    # Write all question information to the file
                    with open(question_file, 'w', encoding='utf-8') as qf:
                        # Write question_id first
                        qf.write(f"question_id: {question_id}\n")
                        qf.write("=" * 50 + "\n\n")
                        
                        # Write question_title
                        question_title = question_data.get('question_title', '')
                        if question_title.strip():
                            qf.write("QUESTION TITLE:\n")
                            qf.write("-" * 20 + "\n")
                            qf.write(question_title + "\n\n")
                        
                        # Write difficulty
                        difficulty = question_data.get('difficulty', '')
                        if difficulty.strip():
                            qf.write("DIFFICULTY:\n")
                            qf.write("-" * 20 + "\n")
                            qf.write(difficulty + "\n\n")
                        
                        # Write question_content
                        qf.write("QUESTION CONTENT:\n")
                        qf.write("-" * 20 + "\n")
                        qf.write(question_data.get('question_content', '') + "\n\n")
                        
                        # Write starter_code if present
                        starter_code = question_data.get('starter_code', '')
                        if starter_code.strip():
                            qf.write("STARTER CODE:\n")
                            qf.write("-" * 20 + "\n")
                            qf.write(starter_code + "\n\n")
                        
                        # Write public_test_cases
                        public_tests = question_data.get('public_test_cases', [])
                        if public_tests:
                            qf.write("PUBLIC TEST CASES:\n")
                            qf.write("-" * 20 + "\n")
                            
                            # Parse public_test_cases if it's a JSON string
                            if isinstance(public_tests, str):
                                try:
                                    public_tests = json.loads(public_tests)
                                except json.JSONDecodeError:
                                    # If parsing fails, treat as raw string
                                    qf.write(f"  {public_tests}\n\n")
                                    continue
                            
                            for i, test_case in enumerate(public_tests, 1):
                                qf.write(f"Test Case {i}:\n")
                                if isinstance(test_case, dict):
                                    # Handle structured test cases
                                    if 'input' in test_case:
                                        input_text = test_case['input']
                                        # Format input with proper line breaks
                                        if '\n' in input_text:
                                            qf.write("  Input:\n")
                                            for line in input_text.split('\n'):
                                                if line.strip():  # Skip empty lines
                                                    qf.write(f"    {line}\n")
                                        else:
                                            qf.write(f"  Input: {input_text}\n")
                                    
                                    if 'output' in test_case:
                                        output_text = test_case['output']
                                        # Format output with proper line breaks
                                        if '\n' in output_text:
                                            qf.write("  Output:\n")
                                            for line in output_text.split('\n'):
                                                if line.strip():  # Skip empty lines
                                                    qf.write(f"    {line}\n")
                                        else:
                                            qf.write(f"  Output: {output_text}\n")
                                    
                                    if 'stdin' in test_case:
                                        qf.write(f"  Stdin: {test_case['stdin']}\n")
                                    if 'stdout' in test_case:
                                        qf.write(f"  Stdout: {test_case['stdout']}\n")
                                    if 'testtype' in test_case:
                                        qf.write(f"  Type: {test_case['testtype']}\n")
                                else:
                                    # Handle simple test cases
                                    qf.write(f"  {test_case}\n")
                                qf.write("\n")
                    
                    questions_in_file += 1
                    
                    # Progress indicator
                    if questions_in_file % 50 == 0:
                        print(f"  Processed {questions_in_file} questions...")
                        
                except json.JSONDecodeError as e:
                    print(f"  Error parsing line {line_num}: {e}")
                except Exception as e:
                    print(f"  Error processing line {line_num}: {e}")
        
        total_questions += questions_in_file
        print(f"  ✅ Created {questions_in_file} individual question files in {test_name}/")
    
    print(f"\n🎉 Successfully created {total_questions} individual question files!")
    print(f"📁 Output directory: {output_path}")
    
    # Print summary by test folder
    print("\n📊 Summary by test folder:")
    for test_dir in sorted(output_path.iterdir()):
        if test_dir.is_dir():
            question_count = len(list(test_dir.glob("*.txt")))
            print(f"  {test_dir.name}: {question_count} questions")

def main():
    """Main function to run the dataset division."""
    input_dir = "data/refined_data"
    output_dir = "data/individual_questions"
    
    # Check if input directory exists
    if not Path(input_dir).exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        print("Please run the refine_dataset.py script first.")
        return
    
    print("🔧 Dividing refined dataset into individual question files...")
    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")
    print()
    
    create_individual_questions(input_dir, output_dir)

if __name__ == "__main__":
    main()
