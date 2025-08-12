import os
import json
from pathlib import Path
from openai import OpenAI

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
]

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def classify_question(question_content: str) -> str:
    """
    Classifies a question into one of the predefined types using OpenAI's API.
    """
    prompt = f"""
    You are an expert in competitive programming problem classification. Your task is to accurately categorize coding challenge questions.

    Given the following question content, classify it into ONE and only ONE of the following predefined types. Choose the best fitting category.

    Available Classification Types:
    {", ".join(CLASSIFICATION_TYPES)}

    Question Content:
    ---
    {question_content}
    ---

    Your response must be the exact name of the chosen classification type from the list above, and nothing else.
    For example, if the question is about sorting, respond with "Searching & Sorting".
    If it involves traversing a tree, respond with "Trees".
    If it's a straightforward problem that requires direct coding of a described process, choose "Implementation".
    If it's about finding optimal substructure and overlapping subproblems, choose "Dynamic Programming (DP)".
    If the problem cannot be clearly categorized into any of the above, consider "Math/Ad-hoc" if it's primarily a mathematical puzzle or a problem with no standard algorithmic solution, or "Unclassified" as a last resort.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",  # You can choose a different model if needed
            temperature=0.0, # Make it deterministic
            max_tokens=50, # Limit token generation to avoid long responses
        )
        classification = chat_completion.choices[0].message.content.strip()
        # Ensure the classification is exactly one of the predefined types
        if classification in CLASSIFICATION_TYPES:
            return classification
        else:
            print(f"Warning: Received an unlisted classification: {classification}. Trying to find the closest match or defaulting to 'Unclassified'.")
            # Attempt to find closest match or assign 'Unclassified' if no close match is found
            # This can be improved with fuzzy matching if needed
            for q_type in CLASSIFICATION_TYPES:
                if q_type.lower() in classification.lower() or classification.lower() in q_type.lower():
                    return q_type
            return "Unclassified"
    except Exception as e:
        print(f"Error classifying question: {e}")
        return "Error"

def process_and_classify_dataset(input_dir: str, output_dir: str):
    """
    Reads JSONL files from input_dir, classifies each question, and writes
    to type-specific JSONL files in output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(exist_ok=True)

    # Prepare output files
    output_files = {}
    # Include "Unclassified" and "Error" in the types to create files for
    all_output_types = CLASSIFICATION_TYPES + ["Unclassified", "Error"]
    for q_type in all_output_types:
        # Sanitize type names for filenames
        filename = q_type.replace('/', '_').replace(' ', '_').replace('-', '_') + ".jsonl"
        output_files[q_type] = open(output_path / filename, 'a', encoding='utf-8')

    jsonl_files = list(input_path.glob("*.jsonl"))
    total_questions = 0
    classified_counts = {q_type: 0 for q_type in all_output_types}

    print(f"Found {len(jsonl_files)} JSONL files to process in {input_dir}...")

    for jsonl_file in jsonl_files[:1]:
        print(f"\nProcessing {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > 50:
                    break
                try:    
                    question_data = json.loads(line.strip())
                    question_content = question_data.get('question_content', '')
                    
                    if not question_content:
                        print(f"  Skipping line {line_num} in {jsonl_file.name}: 'question_content' is empty or missing. Assigning to 'Unclassified'.")
                        classified_type = "Unclassified"
                    else:
                        classified_type = classify_question(question_content)
                    
                    # Write to the corresponding type-specific JSONL file
                    json.dump(question_data, output_files[classified_type], ensure_ascii=False)
                    output_files[classified_type].write('\n')
                    classified_counts[classified_type] += 1
                    total_questions += 1

                    if total_questions % 10 == 0:
                        print(f"  Processed {total_questions} questions so far...")

                except json.JSONDecodeError as e:
                    print(f"  Error parsing line {line_num} in {jsonl_file.name}: {e}. Skipping this line.")
                    # Optionally, log problematic lines to an error file
                except Exception as e:
                    print(f"  An unexpected error occurred processing line {line_num} in {jsonl_file.name}: {e}. Assigning to 'Error'.")
                    classified_type = "Error" # Assign to error type if an unexpected exception occurs
                    try:
                        json.dump(question_data, output_files[classified_type], ensure_ascii=False)
                        output_files[classified_type].write('\n')
                        classified_counts[classified_type] += 1
                        total_questions += 1
                    except Exception as inner_e:
                        print(f"    Further error writing to error file: {inner_e}")


    for f in output_files.values():
        f.close()

    print(f"\n🎉 Successfully processed and classified {total_questions} questions!")
    print(f"📁 Output directory: {output_path}")

    print("\n📊 Classification Summary:")
    for q_type, count in classified_counts.items():
        print(f"  {q_type}: {count} questions")


def main():
    input_dir = "data/refined_data" # Assuming refined data is here, similar to the original script
    output_dir = "data/divided_data"

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
