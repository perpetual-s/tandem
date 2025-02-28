import os
import pyarrow.parquet as pq
import pandas as pd

from tandem_core import get_initial_answer, meta_evaluate_and_refine_answer
import model_manager

def extract_problem_text(row):
    """
    Extracts the math problem text from a DataFrame row.
    It first checks for a 'problem' column; if not found,
    it looks for a 'prompt' column, which is expected to be
    a list of dictionaries (e.g. [{'content': "Problem text", ...}, ...]).
    If the key "content" exists, it returns that value; otherwise,
    it returns the first value from the dict.
    """
    # Prefer a 'problem' column if it exists.
    if 'problem' in row and pd.notnull(row['problem']):
        return row['problem']
    elif 'prompt' in row and pd.notnull(row['prompt']):
        prompt_cell = row['prompt']
        # If prompt_cell is a list and not empty:
        if isinstance(prompt_cell, list) and len(prompt_cell) > 0:
            first_entry = prompt_cell[0]
            if isinstance(first_entry, dict):
                # Try to use a "content" key if present:
                if "content" in first_entry:
                    return first_entry["content"]
                else:
                    # Otherwise, return the first value from the dict
                    return next(iter(first_entry.values()))
            else:
                return str(first_entry)
        else:
            return str(prompt_cell)
    return ""

def process_problem(problem: str) -> str:
    """
    Processes a single math problem:
      1. Prepares a temporary model based on the problem’s classification.
      2. Generates an initial answer.
      3. Iteratively refines the answer.
      4. Cleans up by deleting the temporary model.
      
    Returns the final answer as a string.
    """
    print("\n===================================")
    print(f"Processing problem:\n{problem}\n")
    
    # Step 1: Prepare a temporary model using the problem text.
    temp_model = model_manager.prepare_temporary_model(problem)
    if not temp_model:
        print("Error preparing temporary model for the problem.")
        return ""
    print(f"Using temporary model: {temp_model}")
    
    # Step 2: Generate an initial answer.
    initial_answer = get_initial_answer(problem, model=temp_model)
    print("\nInitial Answer:")
    print(initial_answer)
    
    # Step 3: Iteratively refine the answer.
    final_answer = meta_evaluate_and_refine_answer(initial_answer, problem, model=temp_model, max_iterations=3)
    print("\nFinal Evaluated and Refined Answer:")
    print(final_answer)
    
    # Clean up: Delete the temporary model.
    model_manager.delete_temporary_model(temp_model)
    
    return final_answer

def main():
    test_data_dir = "_test_data"
    if not os.path.exists(test_data_dir):
        print(f"Test data directory '{test_data_dir}' does not exist.")
        return

    # Find all Parquet files in the test data directory.
    parquet_files = [
        os.path.join(test_data_dir, f)
        for f in os.listdir(test_data_dir)
        if f.endswith(".parquet")
    ]
    
    if not parquet_files:
        print("No Parquet files found in the test data directory.")
        return

    # Process each Parquet file.
    for parquet_file in parquet_files:
        print(f"\nReading Parquet file: {parquet_file}")
        try:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")
            continue

        # Determine which column to use: 'problem' or fallback to 'prompt'.
        if "problem" in df.columns:
            col = "problem"
        elif "prompt" in df.columns:
            col = "prompt"
        else:
            print(f"No 'problem' or 'prompt' column found in {parquet_file}. Skipping file.")
            continue

        # Process each row.
        for idx, row in df.iterrows():
            problem_text = extract_problem_text(row)
            if not problem_text:
                print(f"\n--- Row {idx+1}: Could not extract problem text. Skipping row.")
                continue
            print(f"\n--- Problem {idx+1} ---")
            result = process_problem(problem_text)
            if result:
                print(f"\nResult for Problem {idx+1}:\n{result}")
            else:
                print(f"\nProblem {idx+1} could not be processed.")

if __name__ == "__main__":
    main()
