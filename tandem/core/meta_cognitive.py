"""
Meta-cognitive feedback implementation for the Tandem framework.

This module provides functions for evaluating and refining answers through
iterative feedback loops, where the model critiques its own outputs.
"""

from typing import Dict, Any, List
from tandem.utils.common import get_model_response

def meta_evaluate_and_refine_answer(initial_answer: str, problem: str, model: str, max_iterations: int = 3, 
                                  problem_type: str = None, focused_areas: List[str] = None) -> Dict[str, Any]:
    """
    Iteratively evaluate and refine the answer using enhanced meta-cognitive feedback.
    
    Args:
        initial_answer: The starting answer to refine
        problem: The original problem statement
        model: The model to use for evaluation and revision
        max_iterations: Maximum number of feedback loops
        problem_type: Optional type of problem to tailor the evaluation (math, code, essay, etc.)
        focused_areas: Optional specific areas to focus evaluation on
        
    Returns:
        Dictionary containing the final answer and metadata about the refinement process
    """
    current_answer = initial_answer
    improvement_history = []
    feedback_history = []
    
    # Generate specialized prompts based on problem type
    evaluation_focus = ""
    if problem_type:
        if problem_type.lower() == "math":
            evaluation_focus = (
                "Pay special attention to: mathematical accuracy, correct application of formulas, "
                "logical reasoning steps, numerical calculations, and proper mathematical notation."
            )
        elif problem_type.lower() == "coding":
            evaluation_focus = (
                "Pay special attention to: algorithm correctness, code syntax, edge cases, time/space complexity, "
                "and implementation details."
            )
        elif problem_type.lower() == "writing":
            evaluation_focus = (
                "Pay special attention to: clarity of expression, logical flow, strength of arguments, "
                "evidence provided, and rhetorical effectiveness."
            )
        elif problem_type.lower() == "science":
            evaluation_focus = (
                "Pay special attention to: scientific accuracy, proper application of principles, "
                "experimental reasoning, correct use of terminology, and empirical evidence."
            )
    
    # Add custom focus areas if provided
    if focused_areas:
        custom_focus = ", ".join(focused_areas)
        evaluation_focus += f"\nAdditionally, specifically evaluate these aspects: {custom_focus}."
    
    print(f"\n=== Meta-Cognitive Feedback Process ===")
    for iteration in range(max_iterations):
        print(f"\n=== Meta-Evaluation Iteration {iteration + 1}/{max_iterations} ===")
        
        # Construct evaluation prompt with adaptive components
        base_evaluation_prompt = (
            "You are an expert evaluator reviewing the following answer. "
            "Be critical and thorough in your assessment. "
            "Identify specific issues in these categories:\n"
            "1. Factual accuracy - Are there any errors or misunderstandings?\n"
            "2. Reasoning quality - Is the logical flow sound and complete?\n"
            "3. Clarity - Is anything confusing or unclearly explained?\n"
            "4. Completeness - Is anything important missing?\n"
            "5. Presentation - Is the answer well-structured and appropriate?\n"
        )
        
        if evaluation_focus:
            base_evaluation_prompt += f"\n{evaluation_focus}\n"
        
        # Add information about previous iterations for context
        if iteration > 0:
            base_evaluation_prompt += (
                f"\nThis is refinement iteration {iteration+1}. "
                f"Previous feedback identified these issues: {', '.join(feedback_history)}. "
                "Focus on any remaining issues or new issues introduced during revision.\n"
            )
            
        final_evaluation_prompt = (
            f"{base_evaluation_prompt}\n\n"
            f"Problem:\n{problem}\n\nAnswer to evaluate:\n{current_answer}\n\n"
            "Format your evaluation as a numbered list of specific issues. "
            "If the answer is perfect with no issues, respond only with 'No issues found.'\n\nEvaluation:"
        )
        
        # Get evaluation feedback
        feedback, success = get_model_response(final_evaluation_prompt, model=model)
        if not success:
            print("Evaluation failed. Retaining current answer.")
            break
        
        feedback_history.append(f"Iteration {iteration+1}: {feedback[:100]}...")
        
        # If feedback indicates no issues, we are done
        if "no issues" in feedback.lower() or "perfect" in feedback.lower():
            print("\n=== Evaluation Result: No Issues Found ===")
            print("Evaluator found the answer to be correct and complete. Final answer accepted.")
            break
        
        # Extract key issues for structured improvement
        key_issues = []
        for line in feedback.split('\n'):
            if any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '•', '-', '★']):
                key_issues.append(line.strip())
        
        issue_summary = "; ".join(key_issues) if key_issues else feedback
                
        # Enhanced revision prompt that builds on previous iterations
        revision_context = ""
        if iteration > 0:
            revision_context = (
                f"This is revision iteration {iteration+1}. Focus on addressing all remaining issues "
                f"while preserving the improvements made in previous iterations. "
                f"Previous refinements addressed: {', '.join(improvement_history)}.\n"
            )
            
        revision_prompt = (
            "You are an expert reviser. Based on the critical evaluation, revise the answer "
            "to address all identified issues while maintaining a clear, concise, and correct response.\n"
            f"{revision_context}\n"
            f"Problem:\n{problem}\n\n"
            f"Current Answer:\n{current_answer}\n\n"
            f"Critical Evaluation:\n{feedback}\n\n"
            "First plan your approach to revision by identifying what specific improvements you'll make. "
            "Then provide your complete revised answer that incorporates all needed changes.\n\n"
            "Revised Answer:"
        )
        
        # Get the revised answer
        revised_answer, success = get_model_response(revision_prompt, model=model)
        if not success:
            print("Revision failed. Retaining current answer.")
            break
        
        # Extract the revision plan if it exists (helpful for tracking improvements)
        improvement_made = f"Iteration {iteration+1} improvements"
        if ":" in revised_answer.split("\n")[0]:
            # The first few lines might contain the revision plan
            potential_plan = "\n".join(revised_answer.split("\n")[:3])
            if len(potential_plan) < len(revised_answer) / 2:  # Sanity check
                improvement_made = potential_plan
                # Try to extract just the answer part
                if "Revised Answer:" in revised_answer:
                    revised_answer = revised_answer.split("Revised Answer:")[1].strip()
        
        improvement_history.append(improvement_made)
        
        print("\n=== Evaluating Revision Quality ===")
        
        # Evaluate if the revision is actually better
        comparison_prompt = (
            "Compare the original answer and the revised answer for the given problem. "
            "Determine which one is better overall and explain why.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Original Answer:\n{current_answer}\n\n"
            f"Revised Answer:\n{revised_answer}\n\n"
            "Which answer is better? Respond with either 'Original Answer' or 'Revised Answer' "
            "followed by a brief explanation."
        )
        
        comparison_result, success = get_model_response(comparison_prompt, model=model)
        if not success or "original answer" in comparison_result.lower():
            print("\n=== Revision Result: No Improvement ===")
            print("Revision did not improve the answer. Retaining original version.")
            improvement_history.append("No improvement in this iteration")
            # We don't break, as we might still improve in subsequent iterations
        else:
            current_answer = revised_answer
            print("\n=== Revision Result: Improvement Accepted ===")
            print("The revised answer is better and has been accepted.")
    
    # Prepare the results
    print(f"\n=== Meta-Cognitive Feedback Complete ===")
    print(f"Completed {min(iteration + 1, max_iterations)} iterations")
    print(f"Resolved {len(improvement_history)} issues")
    
    results = {
        "final_answer": current_answer,
        "iterations_used": min(iteration + 1, max_iterations),
        "feedback_history": feedback_history,
        "improvement_history": improvement_history,
        "issues_resolved": len(improvement_history),
    }
    
    return results