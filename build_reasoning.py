import pandas as pd
from openai import OpenAI
from pymilvus import MilvusClient
from pydantic import BaseModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import argparse
import sys

class Output(BaseModel):
    problem_understanding: str
    generalized_solution_pattern: str
    common_mistakes: list[str]

def parse_arguments():
    """Parse command line arguments for the build_reasoning script."""
    parser = argparse.ArgumentParser(
        description="Build reasoning memory from LiveCodeBench evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output file arguments
    parser.add_argument(
        "--input_file",
        type=str,
        default="output/GPT-5-Nano/Scenario.codegeneration_1_0.2_eval_all.json",
        help="Path to the input evaluation JSON file"
    )
    parser.add_argument(
        "--failed_rows_file",
        type=str,
        default="output/failed_processing/failed_rows.json",
        help="Path to the failed rows JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/failed_processing_2",
        help="Directory to save failed rows output"
    )
    
    # Model and API configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Model to use for reasoning generation"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="gemini-embedding-001",
        help="Model to use for generating embeddings"
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="GEMINI_API_KEY",
        help="Environment variable name for API key"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="Base URL for the API"
    )
    
    # Processing configuration
    parser.add_argument(
        "--max_threads",
        type=int,
        default=20,
        help="Maximum number of concurrent threads for processing"
    )

    return parser.parse_args()

def get_milvus_client(endpoint, token):
    """Initialize Milvus client with given endpoint and token."""
    if not endpoint or not token:
        raise ValueError("Milvus endpoint and token must be provided")
    
    return MilvusClient(uri=endpoint, token=token)

def get_embedding(text, llm_client, model="gemini-embedding-001"):
    """Generate embedding for given text using specified model."""
    text = text.replace("\n", " ")
    return llm_client.embeddings.create(input=[text], model=model).data[0].embedding

def process_row(i, row, llm_client, db_client, args):
    """Process a single row to generate reasoning memory."""
    try:
        # Validate required columns exist
        required_columns = ["question_content", "code_list", "graded_list", "question_id"]
        for col in required_columns:
            if col not in row:
                raise ValueError(f"Missing required column: {col}")
        
        if not row["code_list"] or len(row["code_list"]) == 0:
            raise ValueError(f"Empty code_list for row {i}")
        
        user_prompt = f"""
        You are a competitive programming assistant tasked with creating a reusable reflection memory for hard problems.
        This memory will be stored in a knowledge base, tagged with metadata, and later retrieved to help solve similar problems.

        Follow these rules:
        - Output in strict JSON format, no extra text outside the JSON.
        - Keep the reasoning self-contained and reusable across different but similar problems.
        - Avoid copying problem-specific values, variable names, or context — replace them with generic placeholders.
        - For algorithms, state the category (e.g., "brute-force with pruning", "dynamic programming with bitmasking") and conditions under which it works.
        - When possible, include optimizations or variations to broaden applicability.

        Here is the problem statement:
        {row["question_content"]}

        Here is the submitted code:
        {row["code_list"][0]}

        Here is the correctness evaluation of the code:
        {row["graded_list"][0]}

        Now produce the JSON reflection memory in the following structure:
        ```json
        "problem_understanding": "<Restate the problem in generic terms, specifying input format, output format, and constraints in abstract form>",
        "generalized_solution_pattern": "<Step-by-step reusable approach, algorithm type, key data structures, when it works well and possible optimizations.>",
        "common_mistakes": ["<List of common pitfalls or misconceptions that may occur when solving similar problems>"]
        ```
        """
        
        response = llm_client.beta.chat.completions.parse(
            model=args.model,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            response_format=Output
        )

        response_json = response.choices[0].message.parsed
        text = response_json.problem_understanding + " " + response_json.generalized_solution_pattern
        embedded = get_embedding(text, llm_client, args.embedding_model)

        memory = {
            "vector": embedded,
            "payload": {
                "id": i, # Moving 'id' into payload as 'row_index'
                "problem_understanding": response_json.problem_understanding,
                "reflection": response_json.generalized_solution_pattern,
                "common_mistakes": response_json.common_mistakes,
                "code": row["code_list"][0],
                "question_id": row["question_id"],
            }
        }
        
        if not args.dry_run:
            db_client.insert(collection_name=args.collection_name, data=[memory])
        
        if args.debug:
            print(f"Successfully processed row {i}")
            
        return True
    except Exception as e:
        if args.debug:
            print(f"Error in row {i}: {e}")
        return False

def main():
    """Main function to orchestrate the reasoning memory building process."""
    args = parse_arguments()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Load data
    print(f"Loading data from {args.input_file}")
    df = pd.read_json(args.input_file)
    print(f"Loaded {len(df)} rows")
    
    # Determine which rows to process
    if args.process_all:
        df_to_process = df
        print("Processing all rows")
    else:
        if not os.path.exists(args.failed_rows_file):
            raise FileNotFoundError(f"Failed rows file not found: {args.failed_rows_file}")
        temp = pd.read_json(args.failed_rows_file)
        failed_indices = temp["index"].tolist()
        df_to_process = df.iloc[failed_indices]
        print(f"Processing {len(df_to_process)} failed rows")
    
    # Apply index filtering if specified
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(df_to_process)
        df_to_process = df_to_process.iloc[start_idx:end_idx]
        print(f"Filtered to rows {start_idx}-{end_idx}: {len(df_to_process)} rows")
    
    # Apply max rows limit if specified
    if args.max_rows is not None:
        df_to_process = df_to_process.head(args.max_rows)
        print(f"Limited to {len(df_to_process)} rows")
    
    # Validate API key
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"{args.api_key_env} environment variable must be set")
    
    # Initialize clients
    print(f"Initializing clients with model: {args.model}")
    llm_client = OpenAI(api_key=api_key, base_url=args.base_url)
    db_client = get_milvus_client(os.getenv("MILVUS_URL"), os.getenv("MILVUS_TOKEN"))
    
    # Process rows
    failed_rows = []
    
    if args.debug:
        print(f"Starting processing with {args.max_threads} threads")
        if args.dry_run:
            print("DRY RUN MODE: No data will be inserted into database")
    
    with ThreadPoolExecutor(args.max_threads) as executor:
        futures = {
            executor.submit(process_row, i, row, llm_client, db_client, args): (i, row) 
            for i, row in df_to_process.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            i, row = futures[future]
            if not future.result():
                failed_rows.append({"index": i, "data": row.to_dict()})
    
    # Save results
    print(f"Finished processing. {len(failed_rows)} rows failed.")
    
    if failed_rows:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "failed_rows.json")
        with open(output_file, 'w') as f:
            json.dump(failed_rows, f, indent=4)
        print(f"Saved {len(failed_rows)} failed rows to {output_file}")
    
    if args.save_intermediate:
        # Save successful processing summary
        summary = {
            "total_processed": len(df_to_process),
            "successful": len(df_to_process) - len(failed_rows),
            "failed": len(failed_rows),
            "model_used": args.model,
            "embedding_model": args.embedding_model,
            "collection_name": "memory"
        }
        summary_file = os.path.join(args.output_dir, "processing_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Saved processing summary to {summary_file}")

if __name__ == "__main__":
    main()
