import os
import json
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal

CLASSIFICATION_TYPES = [
    "Implementation",
    "Greedy Algorithms",
    "Searching & Sorting",
    "Brute Force / Backtracking",
    "Dynamic Programming (DP)",
    "Graph Algorithms",
    "Trees",
    "Number Theory",
    "Combinatorics",
    "Geometry",
    "Bit Manipulation",
    "String Algorithms",
    "Data Structures",
    "Game Theory",
    "Constructive Algorithms",
    "Math/Ad-hoc",
    "Unclassified",
    "Error"
]

ClassificationLiteral = Literal[
    "Implementation",
    "Greedy Algorithms",
    "Searching & Sorting",
    "Brute Force / Backtracking",
    "Dynamic Programming (DP)",
    "Graph Algorithms",
    "Trees",
    "Number Theory",
    "Combinatorics",
    "Geometry",
    "Bit Manipulation",
    "String Algorithms",
    "Data Structures",
    "Game Theory",
    "Constructive Algorithms",
    "Math/Ad-hoc",
    "Unclassified",
    "Error"
]

class Classification(BaseModel):
    classification: ClassificationLiteral

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def classify_question(question_content: str, starter_code: str, public_tests: str) -> str:
    """
    Classifies a question into one of the predefined types using OpenAI's API.
    """

    prompt = f"""
    You are an expert in competitive programming problem classification. Analyze the problem's core algorithmic requirements and solution approach.

    Classification Guidelines:
    
    **Implementation**: Straightforward coding with no complex algorithms - array manipulation, simulation, basic loops
    
    **Greedy Algorithms**: Problems where locally optimal choices lead to global optimum - scheduling, interval problems, minimizing/maximizing with clear greedy choice
    
    **Searching & Sorting**: Binary search, sorting algorithms, finding elements - NOT just using sorted data structures
    
    **Brute Force / Backtracking**: Exhaustive search, trying all possibilities, recursive exploration with pruning
    
    **Dynamic Programming (DP)**: Optimal substructure + overlapping subproblems - memoization, tabulation, optimization problems
    
    **Graph Algorithms**: Explicit graphs, trees as graphs, BFS/DFS, shortest paths, connectivity, topological sort
    
    **Trees**: Tree-specific algorithms, traversals, LCA, tree DP - NOT general graphs
    
    **Number Theory**: Prime numbers, GCD/LCM, modular arithmetic, factorization, mathematical properties
    
    **Combinatorics**: Counting problems, permutations, combinations, inclusion-exclusion, probability
    
    **Geometry**: Coordinate geometry, computational geometry, distances, areas, intersections
    
    **Bit Manipulation**: Bitwise operations are central to the solution approach
    
    **String Algorithms**: Pattern matching, string processing algorithms, text manipulation beyond basic operations
    
    **Data Structures**: Custom data structures, advanced DS usage (segment trees, tries, etc.)
    
    **Game Theory**: Two-player games, optimal strategies, winning/losing positions
    
    **Constructive Algorithms**: Building/constructing solutions step by step, existence proofs
    
    **Math/Ad-hoc**: Mathematical insight, formulas, number patterns, non-standard mathematical reasoning

    Rules:
    1. Pick exactly ONE category.
    2. Focus on the main algorithm needed for the optimal solution.
    3. If multiple apply, choose the most defining one.
    4. Output ONLY the category name from the list — no punctuation, quotes, or extra words.

    Question Content:
    ---
    {question_content}
    ---
    Starter Code:
    ---
    {starter_code}
    ---

    Public Tests:
    ---
    {public_tests}
    ---

    Analyze the problem's core requirements:
    - What is the main algorithmic challenge?
    - What technique would an expert competitive programmer use?
    - What category best describes the solution approach?

    Return exactly one classification from the predefined types above.

    """

    try:
        chat_completion = client.responses.parse(
            model="gpt-5-nano", 
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict single-label classifier. "
                        "Return exactly one label from the allowed list. "
                        "Output ONLY the label string verbatim, no quotes, no punctuation, no explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            reasoning={"effort": "minimal"},
            text_format=Classification,
        )
        classification =  chat_completion.output_parsed.classification
        if classification in CLASSIFICATION_TYPES:
            return classification

        print(f"Warning: Received an unlisted classification: {classification}. Defaulting to 'Unclassified'.")
        return "Unclassified"

    except Exception as e:
        print(f"Error classifying question: {e}")
        return "Error"


def process_and_classify_dataset(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Prepare output files
    output_files = {}
    all_output_types = CLASSIFICATION_TYPES
    for q_type in all_output_types:
        filename = q_type.replace('/', '_').replace(' ', '_').replace('-', '_') + ".jsonl"
        output_files[q_type] = open(output_path / filename, 'a', encoding='utf-8')

    # 🔹 Step 1: Build a set of already processed question IDs
    processed_ids = set()
    for existing_file in output_path.glob("*.jsonl"):
        with open(existing_file, "r", encoding="utf-8") as ef:
            for line in ef:
                try:
                    data = json.loads(line.strip())
                    qid = data.get("question_id")
                    if qid:
                        processed_ids.add(qid)
                except json.JSONDecodeError:
                    continue
    print(f"✅ Found {len(processed_ids)} previously processed questions. They will be skipped.")

    jsonl_files = list(input_path.glob("*.jsonl"))
    total_questions = 0
    classified_counts = {q_type: 0 for q_type in all_output_types}

    print(f"Found {len(jsonl_files)} JSONL files to process in {input_dir}...")

    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    question_data = json.loads(line.strip())
                    qid = question_data.get('question_id')

                    # 🔹 Step 2: Skip if already processed
                    if qid in processed_ids:
                        print(f"  ⏩ Skipping question {qid} (already processed).")
                        continue

                    question_content = question_data.get('question_content', '')
                    question_starter_code = question_data.get('starter_code', '')
                    question_public_tests = question_data.get('public_tests', '')

                    if not question_content:
                        print(f"  Skipping line {line_num}: 'question_content' missing. Assigning to 'Unclassified'.")
                        classified_type = "Unclassified"
                    else:
                        classified_type = classify_question(question_content, question_starter_code, question_public_tests)

                    json.dump(question_data, output_files[classified_type], ensure_ascii=False)
                    output_files[classified_type].write('\n')
                    classified_counts[classified_type] += 1
                    total_questions += 1

                    if total_questions % 10 == 0:
                        print(f"  Processed {total_questions} questions so far...")

                except json.JSONDecodeError as e:
                    print(f"  Error parsing line {line_num}: {e}")
                except Exception as e:
                    print(f"  Unexpected error at line {line_num}: {e}")
                    classified_type = "Error"
                    try:
                        json.dump(question_data, output_files[classified_type], ensure_ascii=False)
                        output_files[classified_type].write('\n')
                        classified_counts[classified_type] += 1
                        total_questions += 1
                    except Exception as inner_e:
                        print(f"    Error writing to error file: {inner_e}")

    for f in output_files.values():
        f.close()

    print(f"\n🎉 Successfully processed and classified {total_questions} new questions!")
    print(f"📁 Output directory: {output_path}")
    print("\n📊 Classification Summary:")
    for q_type, count in classified_counts.items():
        print(f"  {q_type}: {count} questions")

def main():
    input_dir = "data/refined_data"  # Assuming refined data is here, similar to the original script
    output_dir = "data/divided_data_gpt_5_nano"

    if not Path(input_dir).exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        print("Please ensure your refined JSONL files are in this directory.")
        return

    print("🔧 Classifying refined dataset into types...")
    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")
    print()

    process_and_classify_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()
