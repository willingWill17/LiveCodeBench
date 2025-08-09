from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import sys
import os

# Add the path to the lcb_runner directory to sys.path
sys.path.append("/Users/thangtiennguyen/Documents/Cursor/project/LiveCodeBench")

from lcb_runner.runner.custom_evaluator import main as run_evaluation
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

app = FastAPI()

class EvalRequest(BaseModel):
    custom_output_file: str
    scenario: str
    not_fast: bool = False
    release_version: str = "release_latest"
    cot_code_execution: bool = False
    n: int = 10
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2000
    multiprocess: int = 0
    stop: str = "###"
    continue_existing: bool = False
    continue_existing_with_eval: bool = False
    use_cache: bool = False
    cache_batch_size: int = 100
    debug: bool = False
    evaluate: bool = False
    num_process_evaluate: int = 12
    timeout: int = 6
    openai_timeout: int = 90
    tensor_parallel_size: int = -1
    enable_prefix_caching: bool = False
    custom_output_save_name: str | None = None
    dtype: str = "bfloat16"
    start_date: str | None = None
    end_date: str | None = None

@app.post("/evaluate")
async def evaluate_code(request: EvalRequest):
    try:
        # Create a mock args object from the request data
        class MockArgs:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                self.stop = self.stop.split(",")
                if self.tensor_parallel_size == -1:
                    import torch
                    self.tensor_parallel_size = torch.cuda.device_count()
                if self.multiprocess == -1:
                    import os
                    self.multiprocess = os.cpu_count()

        mock_args = MockArgs(**request.dict())
        mock_args.scenario = Scenario(mock_args.scenario) # Convert string to Scenario enum

        # Call the custom_evaluator main function
        # We need to capture stdout to return the metrics
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            run_evaluation(mock_args)
        s = f.getvalue()

        # Assuming run_evaluation prints the metrics at the end
        # We'll need to parse the output, this is a placeholder
        metrics_output = s.strip().split("\n")[-1]
        try:
            metrics = json.loads(metrics_output)
        except json.JSONDecodeError:
            metrics = {"raw_output": metrics_output, "message": "Could not parse metrics from evaluator output"}

        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
