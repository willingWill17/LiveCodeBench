import pandas as pd
from openai import OpenAI
from pymilvus import MilvusClient
from pydantic import BaseModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class Output(BaseModel):
    problem_understanding: str
    generalized_solution_pattern: str
    common_mistakes: list[str]

def get_milvus_client():
    CLUSTER_ENDPOINT = os.getenv("MILVUS_URI")
    TOKEN = os.getenv("MILVUS_TOKEN")
    return MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return llm_client.embeddings.create(input=[text], model=model).data[0].embedding

def process_row(i, row):
    try:
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
        
        response = llm_client.responses.parse(
            model="gpt-5-nano",
            input=[{"role": "user", "content": user_prompt}],
            reasoning={"effort": "medium"},
            text_format=Output,
        )

        response_json = response.output_parsed
        text = response_json.problem_understanding + " " + response_json.generalized_solution_pattern
        embedded = get_embedding(text)

        memory = {
            "id": i,
            "vector": embedded,
            "payload": {
                "problem_understanding": response_json.problem_understanding,
                "reflection": response_json.generalized_solution_pattern,
                "common_mistakes": response_json.common_mistakes,
                "code": row["code_list"][0],
                "question_id": row["question_id"],
            }
        }
        db_client.insert(collection_name="memory", data=[memory])
        return True
    except Exception as e:
        print(f"Error in row {i}: {e}")
        return False

# --- Main ---
df = pd.read_json("output/GPT-5-Nano/Scenario.codegeneration_1_0.2_eval_all.json")    
llm_client = OpenAI()
db_client = get_milvus_client()

max_threads = 20 # Tune based on your rate limit & system
with ThreadPoolExecutor(max_threads) as executor:
    futures = [executor.submit(process_row, i, row) for i, row in df.iterrows()]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Specific Errors"):
        pass

print("Finished processing specific error rows.")
