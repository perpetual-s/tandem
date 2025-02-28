from ollama import chat
from typing import List, Tuple, Callable, Any
import os
import time

# Define available models
MODEL_MATH    = 'qwen2-math:72b'
MODEL_CODE    = 'qwen2.5-coder:32b'
MODEL_REASON  = 'deepseek-r1:70b'   # Heavy, best for reasoning/evaluation (used later for evaluation)
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

# --- Filtration Function ---

def filter_steps(answer: str, desired_steps: int = 5, model: str = MODEL_GENERAL) -> str:
    """
    Condense the given answer into exactly 'desired_steps' bold, high-level steps.
    """
    prompt = (
        "You are an expert filter. The answer below contains too many small, detailed steps. "
        f"Please condense it into exactly {desired_steps} bold, high-level steps. "
        "Remove any redundant or overly detailed parts, and present the result as a numbered list with exactly "
        f"{desired_steps} steps.\n\nOriginal Answer:\n{answer}\n\nFiltered Answer (exactly {desired_steps} steps):"
    )
    filtered_answer, success = get_model_response(prompt, model=model)
    if success:
        return filtered_answer
    else:
        return answer

# --- Divide and Conquer Functions ---

def get_subproblems(problem: str, model: str = MODEL_GENERAL) -> List[str]:
    """
    Break down the problem into clear subproblems using a general-purpose model.
    (Experimental: using MODEL_GENERAL instead of MODEL_REASON for efficiency)
    """
    prompt = (
        "You are an expert problem-solver. Break down the following problem into "
        "clear and distinct subproblems. List each subproblem on a new line.\n\n"
        f"{problem}\n\nSubproblems:"
    )
    response, success = get_model_response(prompt, model=model)
    if success:
        subproblems = [line.strip() for line in response.splitlines() if line.strip()]
        return subproblems if subproblems else [problem]
    else:
        return [problem]

def select_model_for_subproblem(subproblem: str) -> str:
    """
    Choose a specialized model based on keywords in the subproblem.
    """
    lower = subproblem.lower()
    if any(keyword in lower for keyword in ["calculate", "compute", "solve", "math"]):
        return MODEL_MATH
    elif any(keyword in lower for keyword in ["code", "program", "debug"]):
        return MODEL_CODE
    else:
        return MODEL_GENERAL

def aggregate_subanswers(subanswers: List[str], problem: str, model: str = MODEL_GENERAL) -> str:
    """
    Combine individual subanswers into one coherent solution.
    The final answer should be presented in exactly 5 bold, high-level steps.
    """
    aggregation_prompt = (
        "You are an expert aggregator. Combine the following subanswers into a coherent, complete solution "
        "for the overall problem. Present your final answer in exactly 5 bold, high-level steps. "
        "Each step should be concise and capture a major component of the solution without unnecessary details.\n\n"
        f"Problem:\n{problem}\n\nSubanswers:\n" + "\n".join(subanswers) +
        "\n\nAggregated Answer (5 Bold Steps):"
    )
    aggregated_response, success = get_model_response(aggregation_prompt, model=model)
    if success:
        return aggregated_response
    else:
        return "\n".join(subanswers)

def meta_evaluate_and_refine(aggregated_answer: str, problem: str, model: str = MODEL_REASON, max_iterations: int = 3) -> str:
    """
    Iteratively evaluate and refine the aggregated answer using meta-cognitive feedback loops.
    For revisions, instruct the model to produce the final answer as exactly 5 bold, high-level steps.
    """
    current_answer = aggregated_answer
    for iteration in range(max_iterations):
        print(f"\nMeta-Evaluation Iteration {iteration + 1}:")
        evaluation_prompt = (
            "You are an expert evaluator. Evaluate the aggregated solution below for the problem. "
            "List any errors, unclear steps, or areas needing improvement. "
            "If the solution is perfect, simply respond with 'No issues'.\n\n"
            f"Problem:\n{problem}\n\nAggregated Solution:\n{current_answer}\n\nFeedback:"
        )
        feedback, success = get_model_response(evaluation_prompt, model=model)
        if success and "no issues" in feedback.lower():
            print("Evaluator found no issues. Final answer accepted.")
            break
        else:
            revision_prompt = (
                "You are an expert reviser. Based on the following feedback, revise the aggregated solution "
                "to correct any errors or clarify any unclear steps. Provide the final answer in exactly 5 bold, high-level steps.\n\n"
                f"Problem:\n{problem}\n\nCurrent Aggregated Solution:\n{current_answer}\n\nFeedback:\n{feedback}\n\nRevised Answer (5 Bold Steps):"
            )
            revised_answer, success = get_model_response(revision_prompt, model=model)
            if not success:
                print("Revision failed. Retaining current answer.")
                break
            current_answer = revised_answer
    return current_answer

# --- Main Pipeline with Meta-Cognitive Feedback Loops and Filtration Step ---

if __name__ == "__main__":
    # Step 0: Read the overall problem from prompt.txt
    problem = read_prompt_from_file()
    if not problem:
        exit(1)

    print("Original Problem:")
    print(problem)

    # Step 1: Divide the problem into subproblems using the general-purpose model.
    subproblems = get_subproblems(problem, model=MODEL_GENERAL)
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

    # Step 3: Aggregate all subanswers into one preliminary final answer using Llama3.3.
    aggregated_answer = aggregate_subanswers(subanswers, problem, model=MODEL_GENERAL)
    print("\nPreliminary Aggregated Answer (before filtration):")
    print(aggregated_answer)

    # Step 4: Apply the filtration step to force exactly 5 bold, high-level steps.
    filtered_answer = filter_steps(aggregated_answer, desired_steps=5, model=MODEL_GENERAL)
    print("\nFiltered Aggregated Answer (5 Bold Steps):")
    print(filtered_answer)

    # Step 5: Final evaluation and refinement using meta-cognitive feedback loops.
    final_answer = meta_evaluate_and_refine(filtered_answer, problem, model=MODEL_REASON, max_iterations=3)
    
    print("\nFinal Evaluated and Refined Answer (5 Bold Steps):")
    print(final_answer)
