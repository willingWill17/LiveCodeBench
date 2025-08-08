#!/usr/bin/env python3
"""
Analyze LiveCodeBench problem structure and explain all fields needed for LLM solving.
"""

import json
import base64
import zlib
import pickle
from pathlib import Path
from datetime import datetime


def decode_private_test_cases(encoded_data):
    """Decode the private test cases from base64/zlib format."""
    try:
        decoded = json.loads(encoded_data)
        return decoded
    except:
        try:
            decoded = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(encoded_data.encode("utf-8"))
                    )
                )
            )
            return decoded
        except Exception as e:
            return f"Error decoding: {e}"


def analyze_problem_structure():
    """Analyze the structure of a single problem and explain all fields."""
    
    # Load a sample problem
    dataset_path = Path("data/livecodebench_code_generation_lite")
    sample_file = dataset_path / "test.jsonl"
    
    with open(sample_file, 'r') as f:
        first_line = f.readline().strip()
        problem = json.loads(first_line)
    
    print("🔍 LIVE CODE BENCH PROBLEM STRUCTURE ANALYSIS")
    print("=" * 60)
    
    print("\n📋 ALL FIELDS IN A SINGLE PROBLEM:")
    print("-" * 40)
    
    # Analyze each field
    fields_analysis = {
        "question_title": {
            "type": "string",
            "description": "Short title of the programming problem",
            "example": problem["question_title"],
            "required_for_llm": "Yes - helps identify the problem"
        },
        "question_content": {
            "type": "string", 
            "description": "Full problem description including problem statement, input/output format, constraints, and examples",
            "example": problem["question_content"][:200] + "...",
            "required_for_llm": "YES - This is the main problem description"
        },
        "platform": {
            "type": "string",
            "description": "Source platform (leetcode, codeforces, atcoder)",
            "example": problem["platform"],
            "required_for_llm": "Optional - helps understand problem style"
        },
        "question_id": {
            "type": "string",
            "description": "Unique identifier for the problem",
            "example": problem["question_id"],
            "required_for_llm": "No - for tracking only"
        },
        "contest_id": {
            "type": "string", 
            "description": "Contest identifier where the problem appeared",
            "example": problem["contest_id"],
            "required_for_llm": "No - for tracking only"
        },
        "contest_date": {
            "type": "datetime",
            "description": "Date when the contest was held",
            "example": problem["contest_date"],
            "required_for_llm": "No - for tracking only"
        },
        "starter_code": {
            "type": "string",
            "description": "Initial code template provided to the user (can be empty)",
            "example": problem["starter_code"] or "Empty",
            "required_for_llm": "YES - if provided, must be used in solution"
        },
        "difficulty": {
            "type": "string",
            "description": "Problem difficulty level (easy, medium, hard)",
            "example": problem["difficulty"],
            "required_for_llm": "Optional - helps understand complexity"
        },
        "public_test_cases": {
            "type": "list[Test]",
            "description": "Visible test cases with input/output pairs",
            "example": json.loads(problem["public_test_cases"]),
            "required_for_llm": "YES - helps understand expected behavior"
        },
        "private_test_cases": {
            "type": "list[Test]", 
            "description": "Hidden test cases (encoded) used for evaluation",
            "example": "Encoded data (not shown)",
            "required_for_llm": "No - hidden from LLM during generation"
        },
        "metadata": {
            "type": "dict",
            "description": "Additional problem metadata (function names, etc.)",
            "example": json.loads(problem["metadata"]),
            "required_for_llm": "Optional - may contain function names"
        }
    }
    
    for field_name, analysis in fields_analysis.items():
        print(f"\n🔹 {field_name}:")
        print(f"   Type: {analysis['type']}")
        print(f"   Description: {analysis['description']}")
        print(f"   Required for LLM: {analysis['required_for_llm']}")
        if field_name == "public_test_cases":
            test_cases = analysis["example"]
            print(f"   Example test cases:")
            for i, test in enumerate(test_cases):
                print(f"     Test {i+1}:")
                print(f"       Input: {test['input'][:100]}{'...' if len(test['input']) > 100 else ''}")
                print(f"       Output: {test['output'][:100]}{'...' if len(test['output']) > 100 else ''}")
                print(f"       Type: {test['testtype']}")
        elif field_name != "private_test_cases":
            print(f"   Example: {analysis['example']}")
    
    print("\n" + "=" * 60)
    print("🎯 ESSENTIAL FIELDS FOR LLM SOLVING:")
    print("=" * 60)
    
    essential_fields = [
        "question_content",
        "starter_code", 
        "public_test_cases",
        "metadata"
    ]
    
    for field in essential_fields:
        analysis = fields_analysis[field]
        print(f"\n✅ {field}: {analysis['description']}")
    
    print("\n" + "=" * 60)
    print("📝 MINIMAL PROMPT STRUCTURE FOR LLM:")
    print("=" * 60)
    
    print("""
To solve a LiveCodeBench problem, your LLM needs:

1. PROBLEM DESCRIPTION:
   - question_content (full problem statement)

2. CODE TEMPLATE (if provided):
   - starter_code (must be used if not empty)

3. EXAMPLES:
   - public_test_cases (input/output examples)

4. OPTIONAL CONTEXT:
   - metadata (may contain function names)
   - difficulty (for understanding complexity)

The LLM should generate Python code that:
- Reads input from stdin (unless starter_code provided)
- Implements the solution logic
- Outputs results to stdout
- Passes all test cases
""")
    
    print("\n" + "=" * 60)
    print("🔧 TEST CASE STRUCTURE:")
    print("=" * 60)
    
    test_cases = json.loads(problem["public_test_cases"])
    print("Each test case contains:")
    print("- input: string (input data)")
    print("- output: string (expected output)")
    print("- testtype: 'stdin' or 'functional'")
    
    print(f"\nExample from sample problem:")
    for i, test in enumerate(test_cases):
        print(f"Test {i+1}:")
        print(f"  Input: {repr(test['input'])}")
        print(f"  Output: {repr(test['output'])}")
        print(f"  Type: {test['testtype']}")
    
    return problem


if __name__ == "__main__":
    analyze_problem_structure()
