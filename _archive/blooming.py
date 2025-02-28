import random
from typing import List, Tuple
from common_utils import get_model_response

def generate_bloom_candidates(problem: str, model: str, candidate_count: int = 3) -> List[str]:
    """
    Generate multiple candidate answers for the given problem.
    We introduce slight variations in the prompt to encourage diverse outputs.
    """
    candidates = []
    base_prompt = (
        "Please solve the following problem. Provide a detailed and well-structured answer that includes all necessary steps.\n\n"
        f"Problem:\n{problem}\n\nAnswer:"
    )
    
    for i in range(candidate_count):
        # Add a small random variation note to the prompt to diversify responses.
        variation_note = f"\n# Variation note: {random.randint(0, 100)}"
        prompt = base_prompt + variation_note
        candidate, success = get_model_response(prompt, model=model)
        if success:
            candidates.append(candidate)
        else:
            candidates.append("Error generating candidate")
    return candidates

def evaluate_candidates(candidates: List[str], problem: str, model: str) -> List[Tuple[str, float]]:
    """
    Evaluate each candidate answer using a meta-evaluation prompt.
    Each candidate is scored between 0 and 1 (1 indicating a complete, correct, and clear answer).
    Returns a list of tuples: (candidate, score).
    """
    evaluated = []
    for candidate in candidates:
        evaluation_prompt = (
            "You are an expert evaluator. Evaluate the following answer for the problem "
            "and provide a score between 0 and 1, where 1 indicates a complete, correct, and clear answer.\n\n"
            f"Problem:\n{problem}\n\nAnswer:\n{candidate}\n\nScore:"
        )
        score_text, success = get_model_response(evaluation_prompt, model=model)
        try:
            score = float(score_text.strip())
        except ValueError:
            score = 0.0
        evaluated.append((candidate, score))
    return evaluated

def select_best_candidate(evaluated_candidates: List[Tuple[str, float]]) -> str:
    """
    Select the candidate answer with the highest evaluation score.
    """
    if not evaluated_candidates:
        return "No valid candidates generated."
    
    evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
    best_candidate, best_score = evaluated_candidates[0]
    print(f"Best candidate selected with score: {best_score:.2f}")
    return best_candidate

def blooming_solve(problem: str, model: str, candidate_count: int = 3) -> str:
    """
    Implements the blooming method:
      1. Generates multiple candidate answers.
      2. Evaluates each candidate.
      3. Selects and returns the best candidate.
    """
    print("\nGenerating bloom candidates...")
    candidates = generate_bloom_candidates(problem, model, candidate_count)
    
    print("\nEvaluating candidates...")
    evaluated_candidates = evaluate_candidates(candidates, problem, model)
    
    best_candidate = select_best_candidate(evaluated_candidates)
    return best_candidate