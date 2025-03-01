from common_utils import read_prompt_from_file, get_model_response
from typing import Tuple, Dict, Any, List

def get_initial_answer(problem: str, model: str) -> str:
    """
    Generate an initial answer for the problem using detailed chain-of-thought prompting.
    """
    prompt = (
        "Please solve the following problem. Provide a detailed and well-structured answer that includes all necessary steps.\n\n"
        f"Problem:\n{problem}\n\nAnswer:"
    )
    answer, success = get_model_response(prompt, model=model)
    return answer if success else "Error generating answer."

def meta_evaluate_and_refine_answer(initial_answer: str, problem: str, model: str, max_iterations: int = 3, 
                                  problem_type: str = None, focused_areas: List[str] = None,
                                  agreement_percentage: float = None) -> Dict[str, Any]:
    """
    Iteratively evaluate and refine the answer using enhanced meta-cognitive feedback.
    
    Parameters:
    - initial_answer: The starting answer to refine
    - problem: The original problem statement
    - model: The model to use for evaluation and revision
    - max_iterations: Maximum number of feedback loops
    - problem_type: Optional type of problem to tailor the evaluation (math, code, essay, etc.)
    - focused_areas: Optional specific areas to focus evaluation on
    - agreement_percentage: Optional percentage of agreement between solutions
    
    Returns:
    - Dictionary containing the final answer and metadata about the refinement process
    """
    current_answer = initial_answer
    
    # Set minimum iterations based on agreement percentage
    if agreement_percentage is not None and agreement_percentage < 40.0:
        max_iterations = max(5, max_iterations)  # At least 5 iterations for low agreement (<40%)
    elif agreement_percentage is not None and agreement_percentage < 80.0:
        max_iterations = max(3, max_iterations)  # At least 3 iterations for medium agreement (<80%)
        
    improvement_history = []
    feedback_history = []
    critique_categories = {}  # Track critiques by category for targeted improvement
    
    # Get problem-specific critique template
    critique_template, revision_focus = get_problem_specific_templates(problem_type)
    
    print(f"\n\n=== Meta-Cognitive Feedback Process ===\n")
    
    # Stage 1: Initial comprehensive evaluation
    print(f"\n=== Initial Comprehensive Evaluation ===\n")
    evaluation_prompt = create_evaluation_prompt(problem, current_answer, problem_type, focused_areas, 
                                               iteration=0, feedback_history=[], critique_template=critique_template)
    
    feedback, success = get_model_response(evaluation_prompt, model=model)
    if not success:
        print("Initial evaluation failed. Retaining current answer.")
        return {"final_answer": current_answer, "iterations_used": 0, 
                "feedback_history": [], "improvement_history": [], "issues_resolved": 0}
    
    feedback_history.append(f"Initial evaluation: {feedback[:100]}...")
    
    # If feedback indicates no issues, we are done
    if "no issues" in feedback.lower() or "perfect" in feedback.lower():
        print("\n\n=== Evaluation Result: No Issues Found ===")
        print("Evaluator found the answer to be correct and complete. Final answer accepted.\n")
        return {"final_answer": current_answer, "iterations_used": 0, 
                "feedback_history": feedback_history, "improvement_history": [], "issues_resolved": 0}
    
    # Extract and categorize issues from feedback
    critique_categories = extract_critique_categories(feedback)
    
    # Now proceed with targeted iterations based on critique categories
    for iteration in range(max_iterations):
        print(f"\n\n=== Meta-Evaluation Iteration {iteration + 1}/{max_iterations} ===\n")
        
        # Determine the focus for this iteration
        if iteration == 0:
            # First iteration: Address high-priority issues (accuracy and reasoning)
            current_focus = ["accuracy", "reasoning"] 
            print("Focus: Addressing critical accuracy and reasoning issues\n")
        elif iteration == 1:
            # Second iteration: Address completeness and clarity
            current_focus = ["completeness", "clarity"]
            print("Focus: Improving completeness and clarity of the answer\n")
        elif iteration == 2:
            # Third iteration: Address presentation and any remaining issues
            current_focus = ["presentation", "remaining"]
            print("Focus: Refining presentation and addressing remaining issues\n")
        else:
            # Later iterations: Address any remaining issues
            current_focus = ["remaining"]
            print("Focus: Addressing any remaining issues\n")
        
        # Create a targeted revision prompt focused on specific categories
        revision_prompt = create_targeted_revision_prompt(
            problem, current_answer, feedback, 
            current_focus, critique_categories,
            problem_type, revision_focus,
            iteration, improvement_history
        )
        
        # Get the revised answer
        revised_answer, success = get_model_response(revision_prompt, model=model)
        if not success:
            print("Revision failed. Retaining current answer.")
            break
        
        # Record the improvement details
        improvement_made = f"Iteration {iteration+1}: Addressed {', '.join(current_focus)} issues"
        improvement_history.append(improvement_made)
        
        # Update the current answer
        current_answer = revised_answer
        print("\n\n=== Revision Result: Improvement Accepted ===")
        print(f"The answer has been revised focusing on: {', '.join(current_focus)}\n")
        
        # If we're not at the last iteration, re-evaluate the answer for the next iteration
        if iteration < max_iterations - 1:
            print("\n\n=== Re-evaluating for Next Iteration ===\n")
            evaluation_prompt = create_evaluation_prompt(
                problem, current_answer, problem_type, focused_areas,
                iteration + 1, feedback_history, critique_template
            )
            
            feedback, success = get_model_response(evaluation_prompt, model=model)
            if not success:
                print("Re-evaluation failed. Proceeding with current answer.")
                continue
                
            feedback_history.append(f"Iteration {iteration+1} evaluation: {feedback[:100]}...")
            
            # If no more issues, we can stop early
            if "no issues" in feedback.lower() or "perfect" in feedback.lower():
                print("\n\n=== Evaluation Result: No Further Issues Found ===")
                print("Evaluator found the answer to be correct and complete. Final answer accepted.\n")
                break
                
            # Update critique categories for next iteration
            critique_categories = extract_critique_categories(feedback)
    
    # Final verification stage
    print("\n\n=== Final Verification Stage ===\n")
    verification_prompt = (
        "You are an expert verifier. Thoroughly check the following solution for correctness and completeness.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution:\n{current_answer}\n\n"
        "Verify that the solution is correct, complete, clear, and well-presented. "
        "If you find any remaining issues, briefly describe them. Otherwise, confirm the solution is excellent."
    )
    
    verification, success = get_model_response(verification_prompt, model=model)
    if success and "correct" in verification.lower() and not any(error_term in verification.lower() for error_term in ["error", "issue", "incorrect", "mistake", "problem"]):
        print("\nFinal verification passed! The solution is correct and complete.\n")
    else:
        print("\nFinal verification note: There may still be room for improvement in the solution.\n")
    
    # Prepare the results
    print(f"\n\n=== Meta-Cognitive Feedback Complete ===\n")
    print(f"Completed {min(iteration + 1, max_iterations)} iterations")
    print(f"Resolved approximately {len(improvement_history)} issue categories\n")
    
    results = {
        "final_answer": current_answer,
        "iterations_used": min(iteration + 1, max_iterations),
        "feedback_history": feedback_history,
        "improvement_history": improvement_history,
        "issues_resolved": len(improvement_history),
        "final_verification": verification if success else "Verification failed",
    }
    
    return results

def get_problem_specific_templates(problem_type: str = None) -> Tuple[str, str]:
    """
    Get problem-specific critique template and revision focus.
    """
    # Default critique template
    critique_template = (
        "Evaluate the answer across these dimensions:\n"
        "1. Factual accuracy - Identify any errors in facts, calculations, or reasoning\n"
        "2. Reasoning quality - Assess the logical flow and completeness of the reasoning\n"
        "3. Clarity - Check for confusing or unclear explanations\n"
        "4. Completeness - Note any missing important information\n"
        "5. Presentation - Evaluate how well-structured and appropriate the answer is\n"
    )
    
    # Default revision focus
    revision_focus = (
        "Focus your revision on addressing all identified issues while maintaining a clear and correct response."
    )
    
    # Customize based on problem type
    if problem_type:
        if problem_type.lower() == "math":
            critique_template = (
                "Evaluate this math solution across these dimensions:\n"
                "1. Mathematical accuracy - Check for calculation errors, incorrect formulas, or invalid operations\n"
                "2. Solution approach - Assess if the chosen method is appropriate and optimal\n"
                "3. Step-by-step reasoning - Verify each step logically follows from the previous ones\n"
                "4. Completeness - Ensure all parts of the problem are addressed\n"
                "5. Verification - Check if the answer is verified by an alternate method or substitution back\n"
                "6. Notation and presentation - Assess the clarity and correctness of mathematical notation\n"
            )
            revision_focus = (
                "In your revision, focus on mathematical correctness, clear step-by-step reasoning, "
                "and proper verification of the final answer."
            )
            
        elif problem_type.lower() == "coding":
            critique_template = (
                "Evaluate this coding solution across these dimensions:\n"
                "1. Algorithm correctness - Check if the algorithm correctly solves the problem\n"
                "2. Efficiency - Assess time and space complexity\n"
                "3. Implementation - Check for syntax errors, bugs, or incorrect logic\n"
                "4. Edge cases - Verify handling of edge cases and error conditions\n"
                "5. Code style - Evaluate readability, variable names, and code organization\n"
                "6. Documentation - Check if the code is adequately commented and explained\n"
            )
            revision_focus = (
                "In your revision, focus on algorithm correctness, efficiency, proper handling of edge cases, "
                "and clean, well-documented code."
            )
            
        elif problem_type.lower() == "writing":
            critique_template = (
                "Evaluate this written response across these dimensions:\n"
                "1. Content accuracy - Check for factual errors or misrepresentations\n"
                "2. Argument strength - Assess the quality of reasoning and evidence\n"
                "3. Organization - Evaluate the logical flow and structure\n"
                "4. Clarity and style - Check for clear, concise, and effective writing\n"
                "5. Grammar and mechanics - Identify any language errors\n"
                "6. Completeness - Ensure all aspects of the prompt are addressed\n"
            )
            revision_focus = (
                "In your revision, focus on strengthening arguments with evidence, improving organization "
                "and flow, and refining language for clarity and impact."
            )
            
        elif problem_type.lower() == "science":
            critique_template = (
                "Evaluate this scientific explanation across these dimensions:\n"
                "1. Scientific accuracy - Check for factual errors or misunderstandings of concepts\n"
                "2. Reasoning and methodology - Assess the scientific approach and reasoning\n"
                "3. Evidence and data usage - Evaluate how evidence is presented and interpreted\n"
                "4. Terminology - Check for correct use of scientific terms and concepts\n"
                "5. Completeness - Ensure the explanation covers all relevant aspects\n"
                "6. Clarity - Assess how clearly complex scientific ideas are communicated\n"
            )
            revision_focus = (
                "In your revision, focus on scientific accuracy, proper use of evidence, "
                "correct terminology, and clear explanation of complex concepts."
            )
    
    return critique_template, revision_focus

def create_evaluation_prompt(problem: str, answer: str, problem_type: str, focused_areas: List[str], 
                          iteration: int, feedback_history: List[str], critique_template: str) -> str:
    """
    Create a problem-specific evaluation prompt.
    """
    # Start with a base prompt
    base_prompt = (
        "You are an expert evaluator reviewing the following answer. "
        "Be critical and thorough in your assessment.\n\n"
    )
    
    # Add the critique template
    base_prompt += critique_template + "\n\n"
    
    # Add specific focus areas if provided
    if focused_areas:
        custom_focus = ", ".join(focused_areas)
        base_prompt += f"Additionally, specifically evaluate these aspects: {custom_focus}.\n\n"
    
    # Add iteration context for subsequent evaluations
    if iteration > 0:
        base_prompt += (
            f"This is refinement iteration {iteration}. "
            f"Previous feedback identified these issues: {', '.join(feedback_history[-1:])}. "
            "Focus on any remaining issues or new issues introduced during revision.\n\n"
        )
    
    # Complete the prompt with the problem and answer
    final_prompt = (
        f"{base_prompt}"
        f"Problem:\n{problem}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        "Format your evaluation as a numbered list of specific issues, organized by category. "
        "For each issue, explain why it's a problem and suggest how it might be fixed. "
        "If the answer is perfect with no issues, respond only with 'No issues found.'\n\n"
        "Evaluation:"
    )
    
    return final_prompt

def create_targeted_revision_prompt(problem: str, current_answer: str, feedback: str, 
                                 current_focus: List[str], critique_categories: Dict[str, List[str]],
                                 problem_type: str, revision_focus: str,
                                 iteration: int, improvement_history: List[str]) -> str:
    """
    Create a targeted revision prompt focused on specific categories.
    """
    # Extract only the critique points relevant to current focus areas
    focused_feedback = []
    for category in current_focus:
        if category == "remaining":
            # For "remaining", include all categories that haven't been specifically addressed yet
            for cat, points in critique_categories.items():
                if cat not in ["accuracy", "reasoning", "completeness", "clarity", "presentation"] or cat in current_focus:
                    focused_feedback.extend(points)
        elif category in critique_categories:
            focused_feedback.extend(critique_categories[category])
    
    # Create a condensed, focused feedback string
    if focused_feedback:
        condensed_feedback = "Issues to address in this revision:\n" + "\n".join(focused_feedback)
    else:
        # If we don't have categorized feedback, use the original feedback
        condensed_feedback = feedback
    
    # Add context about the current iteration
    revision_context = ""
    if iteration > 0:
        revision_context = (
            f"This is revision iteration {iteration+1}, focusing specifically on {', '.join(current_focus)}. "
            f"In previous iterations, we have addressed: {', '.join(improvement_history)}.\n"
            "Keep the improvements from previous iterations while addressing the current focus areas.\n\n"
        )
    
    # Create the final prompt
    revision_prompt = (
        "You are an expert reviser helping to improve a solution through targeted refinement.\n\n"
        f"{revision_context}"
        f"{revision_focus}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Current Answer:\n{current_answer}\n\n"
        f"Critical Feedback to Address:\n{condensed_feedback}\n\n"
        "First, plan your approach to revision by identifying what specific improvements you'll make. "
        "Then provide your complete revised answer that incorporates all needed changes. "
        "Be sure to keep any correct parts of the original answer while improving the problematic areas.\n\n"
        "Ensure your revised answer:\n"
        "1. Addresses all the issues in the feedback\n"
        "2. Maintains a clear, logical structure\n"
        "3. Is complete and thorough\n"
        "4. Clearly marks the final answer as 'FINAL_ANSWER: [your answer]'\n\n"
        "Revised Answer:"
    )
    
    return revision_prompt

def extract_critique_categories(feedback: str) -> Dict[str, List[str]]:
    """
    Extract and categorize critique points from feedback.
    """
    categories = {
        "accuracy": [],
        "reasoning": [],
        "clarity": [],
        "completeness": [],
        "presentation": [],
        "other": []
    }
    
    # Define category keywords to match against
    category_keywords = {
        "accuracy": ["accuracy", "error", "incorrect", "mistake", "wrong", "false", "calculation", "formula"],
        "reasoning": ["reasoning", "logic", "approach", "method", "strategy", "step", "justification", "why"],
        "clarity": ["clarity", "unclear", "confusing", "ambiguous", "vague", "explain", "explanation"],
        "completeness": ["completeness", "missing", "incomplete", "omitted", "additional", "more", "further"],
        "presentation": ["presentation", "structure", "organization", "format", "layout", "style", "notation"]
    }
    
    # Process each line in the feedback
    for line in feedback.split('\n'):
        line = line.strip()
        if not line or line.lower() == "no issues found.":
            continue
            
        # Check if this is a numbered point or bullet point
        is_point = any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '•', '-', '★', '*'])
        
        if is_point:
            # Determine which category this point belongs to
            categorized = False
            line_lower = line.lower()
            
            for category, keywords in category_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    categories[category].append(line)
                    categorized = True
                    break
                    
            if not categorized:
                categories["other"].append(line)
    
    # If no structured points were found, add the whole feedback to "other"
    if all(len(points) == 0 for points in categories.values()):
        categories["other"].append(feedback)
    
    return categories