from ollama import chat
from typing import List, Tuple, Callable, Any
import os
import time

# Define available models
MODEL_MATH    = 'qwen2-math:72b'
MODEL_CODE    = 'qwen2.5-coder:32b'
MODEL_REASON  = 'deepseek-r1:70b'   # Heavy, best for reasoning/evaluation
MODEL_GENERAL = 'llama3.3:70b'      # Lighter general-purpose model

def read_prompt_from_file(filename: str = "prompt.txt") -> str:
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return ""

def time_function_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_model_response(prompt: str, model: str) -> Tuple[str, bool]:
    try:
        response_chunks: List[str] = []
        model_display_name = model.split(':')[0].upper()
        print(f"\n{model_display_name} Responding:")
        print("-" * 40)
        
        def chat_with_model():
            stream = chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            for chunk in stream:
                content = chunk['message']['content']
                response_chunks.append(content)
                print(content, end='', flush=True)
            return ''.join(response_chunks)
        
        full_response, elapsed_time = time_function_call(chat_with_model)
        print("\n" + "-" * 40)
        print(f"{model_display_name} took {elapsed_time:.2f} seconds to respond.")
        
        return full_response, True
    except Exception as e:
        print(f"Error occurred with {model}: {e}")
        return "", False

# --- Divide and Conquer Functions ---

def get_subproblems(problem: str, model: str = MODEL_REASON) -> List[str]:
    """
    Use Deepseek to break down the problem into clear, distinct subproblems.
    This leverages Deepseek's reasoning strength once, avoiding multiple heavy calls.
    """
    prompt = (
        "You are an expert problem-solver. Break down the following problem into "
        "clear and distinct subproblems. List each subproblem on a new line.\n\n"
        f"{problem}\n\nSubproblems:"
    )
    response, success = get_model_response(prompt, model=model)
    if success:
        # Parse the response by newlines; adjust parsing if needed.
        subproblems = [line.strip() for line in response.splitlines() if line.strip()]
        return subproblems if subproblems else [problem]
    else:
        return [problem]

def select_model_for_subproblem(subproblem: str) -> str:
    """
    Select a specialized model based on keywords within the subproblem.
    Adjust the heuristics as needed.
    """
    lower = subproblem.lower()
    if any(keyword in lower for keyword in ["calculate", "compute", "solve", "math"]):
        return MODEL_MATH
    elif any(keyword in lower for keyword in ["code", "program", "debug"]):
        return MODEL_CODE
    else:
        # For general language or explanation tasks
        return MODEL_GENERAL

def aggregate_subanswers(subanswers: List[str], problem: str, model: str = MODEL_GENERAL) -> str:
    """
    Combine the individual subanswers into one aggregated solution.
    Here we use a lighter model to keep latency down.
    """
    aggregation_prompt = (
        "You are an expert aggregator. Combine the following subanswers into a coherent, complete solution "
        "for the overall problem. Ensure that the final answer flows logically.\n\n"
        f"Problem:\n{problem}\n\nSubanswers:\n" + "\n".join(subanswers) +
        "\n\nAggregated Answer:"
    )
    aggregated_response, success = get_model_response(aggregation_prompt, model=model)
    if success:
        return aggregated_response
    else:
        # Fallback: simply join the subanswers
        return "\n".join(subanswers)

# --- Main Pipeline with Minimal Deepseek Calls ---

if __name__ == "__main__":
    # Read the overall problem from prompt.txt
    problem = read_prompt_from_file()
    if not problem:
        exit(1)

    # Step 1: Divide the problem into subproblems using Deepseek (reasoning model).
    subproblems = get_subproblems(problem, model=MODEL_REASON)
    print("\nIdentified Subproblems:")
    for idx, sub in enumerate(subproblems, 1):
        print(f"{idx}. {sub}")

    # Step 2: Solve each subproblem using the appropriate specialized model.
    subanswers = []
    for idx, sub in enumerate(subproblems, 1):
        chosen_model = select_model_for_subproblem(sub)
        print(f"\nSolving Subproblem {idx} using {chosen_model.split(':')[0]}:")
        sub_prompt = f"Please solve the following subproblem:\n\n{sub}"
        answer, success = get_model_response(sub_prompt, model=chosen_model)
        if success:
            subanswers.append(answer)
        else:
            subanswers.append(f"Error solving subproblem {idx}.")

    # Step 3: Aggregate all subanswers into one preliminary final answer using a lighter model.
    aggregated_answer = aggregate_subanswers(subanswers, problem, model=MODEL_GENERAL)
    print("\nPreliminary Aggregated Answer:")
    print(aggregated_answer)

    # Step 4: Final evaluation and refinement using Deepseek.
    final_evaluation_prompt = (
        "You are an expert evaluator. Evaluate the aggregated solution below for the problem. "
        "Highlight any errors or unclear steps, and then produce a final, refined answer.\n\n"
        f"Problem:\n{problem}\n\nAggregated Solution:\n{aggregated_answer}\n\nFinal Answer:"
    )
    final_answer, success = get_model_response(final_evaluation_prompt, model=MODEL_REASON)
    if not success:
        final_answer = aggregated_answer  # Fallback if evaluation fails

    print("\nFinal Evaluated Answer:")
    print(final_answer)
