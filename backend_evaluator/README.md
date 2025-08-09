# LiveCodeBench Backend Evaluator

This directory contains a FastAPI application that serves as a backend for the LiveCodeBench custom evaluator. It allows you to trigger evaluations via POST requests, avoiding the need to load the entire benchmark into memory for each run.

## Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd LiveCodeBench/backend_evaluator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -e ..
    ```

## Running the Application

To start the FastAPI server, run the following command from within the `backend_evaluator` directory:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://localhost:8000`.

## API Endpoint

### `POST /evaluate`

Triggers an evaluation using the `custom_evaluator.py` script.

**Request Body (JSON):**

All parameters from `lcb_runner/runner/parser.py`'s `get_args` function are accepted, with `custom_output_file` and `scenario` being mandatory.

Example:

```json
{
  "custom_output_file": "/path/to/your/LiveCodeBench/data/individual_solution/Bit_Manipulation/2730.json",
  "scenario": "codegeneration",
  "model": "gpt-3.5-turbo-0301",
  "temperature": 0.2
}
```

**Response (JSON):**

```json
{
  "status": "success",
  "metrics": {
    "pass@1": "...",
    "raw_output": "...",
    "message": "..."
  }
}
```

If the evaluation encounters an error, a 500 HTTP status code will be returned with an error message in the `detail` field.
