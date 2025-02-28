from typing import List, Dict, Tuple, Any
import re
from common_utils import get_model_response

def classify_problem(problem: str, model: str) -> str:
    """
    Classify the problem type to help with specialized evaluation during meta-cognitive feedback.
    Returns a category like "math", "coding", "writing", "science", etc.
    """
    classification_prompt = (
        "Classify the following problem into exactly ONE of these categories: "
        "math, coding, writing, science, history, philosophy, business, trivia, puzzles, other.\n\n"
        f"Problem:\n{problem}\n\n"
        "Respond with only the category name, in lowercase."
    )
    
    category, success = get_model_response(classification_prompt, model=model)
    if not success:
        return None
    
    # Clean up and normalize the response
    category = category.lower().strip()
    
    # Map similar categories 
    category_mapping = {
        "mathematics": "math", 
        "computational": "coding",
        "programming": "coding",
        "algorithm": "coding",
        "essay": "writing",
        "physics": "science",
        "chemistry": "science",
        "biology": "science",
        "economics": "business",
        "finance": "business",
        "management": "business",
        "riddle": "puzzles"
    }
    
    # Apply mapping if needed
    for key, value in category_mapping.items():
        if key in category:
            return value
    
    # Return the first word if multiple were returned
    return category.split()[0] if category else None

def generate_diverse_answers(problem: str, model: str, num_answers: int = 5, problem_category: str = None) -> List[str]:
    """
    Generate multiple diverse answers using different system prompts/approaches.
    Each approach encourages a different reasoning style.
    
    Parameters:
    - problem: The problem to solve
    - model: The model to use
    - num_answers: Number of solutions to generate
    - problem_category: Optional category override (not needed for automatic detection)
    """
    # Import inside function to avoid circular imports
    from model_manager import get_additional_parameters, create_temporary_model, create_temporary_modelfile, delete_temporary_model, ORIGINAL_MODELFIL_PATH, classify_problem
    
    # Use the model_manager's classify_problem function for consistent classification 
    # This ensures we use the same category detection as the model_manager
    problem_category = classify_problem(problem)
    print(f"\n=== Problem Classification ===")
    print(f"Detected problem category: {problem_category}")
    
    # Choose solver parameters based on problem type
    if problem_category and problem_category.lower() in ['math', 'puzzles']:
        # Use math-specific parameters for math problems (lower temperature, more precise)
        solver_params = get_additional_parameters("math")
        solver_model_name = "temp_math_solver_model"
        print(f"\n=== Model Configuration ===")
        print(f"Detected math problem - using precise math parameters with low temperature")
    else:
        # Use standard self-consistency parameters for other problems
        solver_params = get_additional_parameters("self_consistency_solver")
        solver_model_name = "temp_self_consistency_solver_model"
        print(f"\n=== Model Configuration ===")
        print(f"Using diverse solution parameters")
    
    # Create the specialized model
    solver_modelfile = create_temporary_modelfile(ORIGINAL_MODELFIL_PATH, solver_params)
    create_temporary_model(solver_model_name, solver_modelfile)
    print(f"Created specialized model: {solver_model_name}")
    
    # For math problems, use special prompts that emphasize different approaches
    if problem_category and problem_category.lower() == 'math':
        # Use math-focused prompts
        prompts = [
            # Standard step-by-step with verification
            f"Solve this math problem step-by-step, showing all calculations clearly. After finding the solution, verify your answer by checking your work.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Algebraic approach
            f"Solve this problem using algebra. Set up equations, solve them step-by-step, and simplify to find the final answer. Show all your work clearly.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Numerical/computational approach
            f"Approach this as a computational problem. Break it down into clear numerical steps, perform calculations precisely, and verify the result.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Conceptual understanding approach
            f"Begin by understanding the core mathematical concepts in this problem. Identify the key principles, then apply them systematically to reach the solution.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Visual/diagram approach (if applicable)
            f"For this problem, consider using diagrams or visual representations if helpful. Solve step-by-step, explaining your reasoning at each stage.\n\nProblem:\n{problem}\n\nSolution:"
        ]
    else:
        # Use general prompts for non-math problems
        prompts = [
            # Standard approach
            f"Please solve the following problem step by step.\n\nProblem:\n{problem}\n\nSolution:",
            
            # First principles approach
            f"Break down this problem from first principles. Start with the core concepts and build up the solution systematically.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Alternative perspectives approach
            f"Consider this problem from multiple angles. Try to identify at least two different ways to solve it, then choose the most effective approach.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Detailed verification approach
            f"Solve this problem step by step. After finding your solution, carefully verify each step to check for correctness.\n\nProblem:\n{problem}\n\nSolution:",
            
            # Complex decomposition approach
            f"Decompose this problem into smaller, manageable parts. Solve each part step by step, then integrate the solutions.\n\nProblem:\n{problem}\n\nSolution:"
        ]
    
    # Use additional prompts if we need more than 5 answers
    if num_answers > 5:
        if problem_category and problem_category.lower() == 'math':
            # Additional math-specific prompts
            extra_math_prompts = [
                f"Solve this using an alternative method than you might typically use. Think outside the box while maintaining mathematical rigor.\n\nProblem:\n{problem}\n\nSolution:",
                f"Break this problem into smaller sub-problems. Solve each component separately, then combine the results for the final answer.\n\nProblem:\n{problem}\n\nSolution:",
                f"Start by identifying potential pitfalls or common mistakes in this type of problem. Then solve carefully, avoiding these issues.\n\nProblem:\n{problem}\n\nSolution:"
            ]
            prompts.extend(extra_math_prompts)
        else:
            # Additional general prompts
            extra_prompts = [
                f"Approach this problem methodically, explaining your reasoning at each step. Consider edge cases and alternative interpretations.\n\nProblem:\n{problem}\n\nSolution:",
                f"Before diving into calculations, analyze what this problem is really asking. Then provide a clear, step-by-step solution.\n\nProblem:\n{problem}\n\nSolution:",
                f"Imagine explaining this problem to someone who doesn't understand it. Break it down clearly and solve it step by step.\n\nProblem:\n{problem}\n\nSolution:"
            ]
            prompts.extend(extra_prompts)
    
    # Use only the number of prompts we need
    prompts = prompts[:num_answers]
    
    answers = []
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating solution {i+1}/{num_answers}...")
        # Use the specialized solver model instead of the original model
        answer, success = get_model_response(prompt, model=solver_model_name)
        if success:
            answers.append(answer)
        else:
            print(f"Failed to generate solution {i+1}")
    
    # Clean up the temporary model
    delete_temporary_model(solver_model_name)
    print(f"\n=== Model Cleanup ===")
    print(f"Deleted specialized solver model: {solver_model_name}")
    
    return answers

def normalize_math_answer(answer: str) -> str:
    """
    Normalizes mathematical answers by removing unnecessary formatting and standardizing the format.
    This helps with consistent comparison between different representations of the same answer.
    """
    # Only process if we have a valid answer
    if not answer or not answer.strip():
        return answer
    
    # Remove surrounding parentheses if they enclose the entire answer
    if answer.startswith('(') and answer.endswith(')'):
        answer = answer[1:-1].strip()
    
    # Remove "x =" prefix that might appear in equations
    answer = re.sub(r'^[a-zA-Z]\s*=\s*', '', answer)
    
    # Remove "=" if it starts the answer
    answer = re.sub(r'^=\s*', '', answer)
    
    # Remove common prefixes like "is", "equals", etc.
    answer = re.sub(r'^(?:is|equals|equal to|is equal to)\s*', '', answer)
    
    # Normalize spaces in math expressions
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Remove trailing punctuation
    answer = re.sub(r'[.,;:]+$', '', answer).strip()$', '', answer).strip()
    
    # Remove "is" or "equals" if it's a separate word at the beginning
    answer = re.sub(r'^(?:is|equals)\s+', '', answer).strip()
    
    # Remove surrounding brackets of all types
    if (answer.startswith('(') and answer.endswith(')')) or \
       (answer.startswith('[') and answer.endswith(']')) or \
       (answer.startswith('{') and answer.endswith('}')):
        answer = answer[1:-1].strip()
    
    # Try to extract just the number if the answer contains a number
    number_match = re.search(r'(\d+(?:\.\d+)?)', answer)
    if number_match and len(number_match.group(1)) > 0 and len(answer) > len(number_match.group(1)) + 3:
        # If the numeric part is significant and there's a lot of extra text,
        # consider just using the numeric part
        if not any(x in answer for x in ['+', '-', '*', '/', '^', 'sqrt']):  # Not a complex expression
            answer = number_match.group(1)
    
    return answer

def extract_final_answer(full_solution: str, problem_type: str = None) -> str:
    """
    Extract just the final answer from a full solution.
    Uses problem type-specific patterns when available.
    """
    # Look for common patterns that indicate final answers based on problem type
    if problem_type and problem_type.lower() == 'math':
        # Math-specific patterns
        math_patterns = [
            # LaTeX boxed answers (common in mathematical formatting)
            r"\\\boxed\{(.*?)\}",
            
            # Expressions like "The answer is x" or "= x" with LaTeX formatting
            r"(?:final answer|the answer is|therefore|thus|hence|equals)[:\s]+(?:\$\\boxed\{(.*?)\}\$|\$(.*?)\$|([\d\.\-\+\/\*x\^≈π√\(\)]+))(?:\.|$)",
            
            # Basic equality with potential LaTeX
            r"[=]\s*(?:\$\\boxed\{(.*?)\}\$|\$(.*?)\$|([\d\.\-\+\/\*x\^≈π√\(\)]+))",
            
            # Numerical answers, possibly with units
            r"(?:\n|^)[^\n]*?(?:=|equals)\s*(?:\$\\boxed\{(.*?)\}\$|\$(.*?)\$|([\d\.\-\+\/\*x\^≈π√\(\)]+(?:\s*[a-zA-Z²³]+)?))",
            
            # Common conclusion phrases 
            r"(?:final result|value of|answer)[:\s]+(?:\$\\boxed\{(.*?)\}\$|\$(.*?)\$|([\d\.\-\+\/\*x\^≈π√\(\)]+))",
            
            # Any individual boxed expressions or formatted answers
            r"is:?\s*\$\\boxed\{(.*?)\}\$",
            r"answer\s+is:?\s*\$(.*?)\$",
            r"answer\s+is:?\s*([\d\.\-\+\/\*x\^≈π√\(\)]+)",
            r"final answer:?\s*\$(.*?)\$",
            r"final answer:?\s*([\d\.\-\+\/\*x\^≈π√\(\)]+)"
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, full_solution, re.IGNORECASE)
            if matches:
                # Clean up the answer
                if isinstance(matches[-1], tuple):
                    # Some patterns might capture multiple groups, pick the first non-empty one
                    for group in matches[-1]:
                        if group:
                            answer = group.strip()
                            break
                    else:
                        # Fallback if no group is non-empty
                        answer = str(matches[-1]).strip()
                else:
                    answer = matches[-1].strip()
                
                # Process any LaTeX boxed content
                if answer.startswith('\\boxed{') and answer.endswith('}'):
                    answer = answer[7:-1]  # Remove \boxed{...}
                
                # Clean up any dollar signs
                answer = answer.replace('$', '').strip()
                
                # Normalize mathematical answers
                # Remove surrounding parentheses if they enclose the entire answer
                if answer.startswith('(') and answer.endswith(')'):
                    answer = answer[1:-1].strip()
                
                # Remove "x =" prefix that might appear in equations
                answer = re.sub(r'^[a-zA-Z]\s*=\s*', '', answer)
                
                # Remove "=" if it starts the answer
                answer = re.sub(r'^=\s*', '', answer)
                
                # Remove common prefixes like "is", "is equal to", etc.
                answer = re.sub(r'^(?:is|equals|equal to|is equal to)\s*', '', answer)
                
                # Normalize spaces in math expressions
                answer = re.sub(r'\s+', ' ', answer).strip()
                
                # Remove trailing punctuation
                answer = re.sub(r'[.,;:]+$', '', answer).strip()
                
                # For math problems, try to keep only the essential numerical/algebraic part
                print(f"Normalized answer: '{answer}'")  # Debug output
                return answer
    
    # General patterns for all problem types
    general_patterns = [
        r"(?:final answer|the answer is|therefore,?|thus,?|so,?|hence,?)[:\s]+(.*?)(?:\.|$)",
        r"(?:conclusion|in conclusion|to summarize)[:\s]+(.*?)(?:\.|$)",
        r"(?:\n)(?:answer|result)[:\s]+(.*?)(?:\.|$)",
        r"[\=]\s*([\d\.\-]+)"  # For numerical answers
    ]
    
    for pattern in general_patterns:
        matches = re.findall(pattern, full_solution, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Return the last match
    
    # If no patterns match, try these fallback approaches
    
    # 1. Look for the last line that contains an equation or answer-like content
    lines = full_solution.strip().split('\n')
    for line in reversed(lines):
        if '=' in line or any(marker in line.lower() for marker in ['answer', 'result', 'therefore', 'thus']):
            return line.strip()
    
    # 2. If still no match, return the last non-empty line
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        return non_empty_lines[-1].strip()
    
    return full_solution  # Fallback to the full solution

def select_answer_by_voting(answers: List[str], problem: str, model: str, problem_category: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    Select the best answer by extracting final answers and voting.
    Returns the most common final answer along with metadata.
    """
    if not answers:
        return "No solutions were generated successfully.", {"error": "No solutions generated"}
        
    # We already have the problem category from the parent function
    
    
    print("\n=== Extracting Final Answers for Voting ===")
    final_answers = []
    for i, full_solution in enumerate(answers):
        # Extract the final answer using regex patterns
        raw_answer = extract_final_answer(full_solution, problem_category)
        
        # Normalize the answer for consistent comparison
        if problem_category and problem_category.lower() == 'math':
            normalized_answer = normalize_math_answer(raw_answer)
        else:
            normalized_answer = raw_answer
            
        final_answers.append(normalized_answer)
        print(f"Solution {i+1} final answer: {normalized_answer} (extracted: '{raw_answer}')")
    
    # Count the occurrences of each answer
    answer_counts = {}
    for answer in final_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    # Find the answer with the most votes
    sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_answer = sorted_answers[0][0]
    most_common_count = sorted_answers[0][1]
    
    # Find which full solution contains this answer
    for i, full_solution in enumerate(answers):
        if extract_final_answer(full_solution) == most_common_answer:
            best_solution_index = i
            break
    else:
        best_solution_index = 0  # Fallback
    
    # Create metadata
    metadata = {
        "answer_counts": answer_counts,
        "most_common_answer": most_common_answer,
        "most_common_count": most_common_count,
        "total_solutions": len(answers),
        "agreement_percentage": (most_common_count / len(answers)) * 100 if answers else 0,
        "best_solution_index": best_solution_index,
        "problem_category": problem_category
    }
    
    # Get the full solution that produced the most common answer
    best_full_solution = answers[best_solution_index]
    
    confidence_description = ""
    if metadata["agreement_percentage"] > 80:
        confidence_description = "Very high confidence"
    elif metadata["agreement_percentage"] > 60:
        confidence_description = "High confidence"
    elif metadata["agreement_percentage"] > 40:
        confidence_description = "Moderate confidence"
    else:
        confidence_description = "Low confidence"
    
    print(f"\n=== Voting Results ===")
    print(f"Selected answer with {metadata['agreement_percentage']:.1f}% agreement ({confidence_description})")
    print(f"Problem category: {problem_category if problem_category else 'Not classified'}")
    
    # Print all extracted answers for debugging
    print("\n=== Answer Distribution ===")
    for answer, count in answer_counts.items():
        print(f"Answer: '{answer}' - Count: {count}")
    
    return best_full_solution, metadata

def verify_answer_correctness(selected_answer: str, problem: str, model: str) -> Tuple[bool, str]:
    """
    Use a specialized verification model to verify the correctness of the selected answer.
    Returns a tuple of (is_correct, feedback).
    """
    # Import here to avoid circular imports
    from model_manager import get_additional_parameters, create_temporary_model, create_temporary_modelfile, delete_temporary_model, ORIGINAL_MODELFIL_PATH
    
    # Create a specialized model for verification with optimized parameters
    verification_params = get_additional_parameters("verification")
    verification_modelfile = create_temporary_modelfile(ORIGINAL_MODELFIL_PATH, verification_params)
    verification_model_name = "temp_verification_model"
    
    create_temporary_model(verification_model_name, verification_modelfile)
    print(f"\n=== Verification Setup ===")
    print(f"Created specialized model for verification: {verification_model_name}")
    
    verification_prompt = (
        "You are an expert evaluator. Verify the correctness of the following solution to the problem.\n"
        "Carefully check each step and the final answer. If there are any errors, incorrect reasoning, "
        "or computational mistakes, point them out specifically. If the solution is correct, state so explicitly.\n\n"
        "Be extremely thorough and critical in your assessment. Check:\n"
        "1. Mathematical accuracy (calculations, formulas, operations)\n"
        "2. Logical reasoning (valid steps, no logical errors)\n"
        "3. Completeness (all parts of the problem addressed)\n"
        "4. Final answer correctness (correct value, format, and units)\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution:\n{selected_answer}\n\n"
        "Is this solution correct? Provide detailed feedback:"
    )
    
    feedback, success = get_model_response(verification_prompt, model=verification_model_name)
    
    # Check if the feedback indicates the solution is correct
    is_correct = any(phrase in feedback.lower() for phrase in ["solution is correct", "correct solution", "answer is correct", "reasoning is correct"])
    
    # Clean up the temporary model
    delete_temporary_model(verification_model_name)
    print(f"\n=== Verification Complete ===")
    print(f"Deleted verification model: {verification_model_name}")
    print(f"Verification result: {'Correct' if is_correct else 'Issues found'}")
    
    return is_correct, feedback

def self_consistency_solve(problem: str, model: str, num_answers: int = 5, 
                       use_meta_cognitive: bool = True, confidence_threshold: float = 60.0,
                       meta_max_iterations: int = 3) -> Dict[str, Any]:
    """
    Main function to solve a problem using the self-consistency approach.
    
    Parameters:
    - problem: The problem to solve
    - model: The model to use
    - num_answers: Number of different solutions to generate
    - use_meta_cognitive: Whether to use meta-cognitive feedback for low confidence solutions
    - confidence_threshold: Below this threshold (in percentage), apply meta-cognitive refinement
    - meta_max_iterations: Maximum number of meta-cognitive feedback loops
    
    Returns a dictionary containing:
    - solution: The selected best solution
    - metadata: Information about the voting process
    - verification: Results of verification
    - meta_cognitive_applied: Whether meta-cognitive refinement was applied
    """
    print(f"\n=== Solving using self-consistency with {num_answers} solutions ===")
    
    # Generate diverse solutions with automatic classification 
    diverse_answers = generate_diverse_answers(problem, model, num_answers)
    
    # Get the problem category from the first function to use in later steps
    from model_manager import classify_problem
    problem_category = classify_problem(problem)
    if not diverse_answers:
        return {
            "solution": "Failed to generate any solutions.",
            "metadata": {"error": "No solutions generated", "problem_category": problem_category},
            "verification": {"is_correct": False, "feedback": "No solutions to verify"},
            "meta_cognitive_applied": False
        }
    
    # Select the best answer through voting
    selected_answer, metadata = select_answer_by_voting(diverse_answers, problem, model, problem_category)
    
    # Determine confidence level
    confidence = metadata.get("agreement_percentage", 0)
    meta_cognitive_applied = False
    meta_iterations_used = 0
    
    # If confidence is low and meta-cognitive feedback is enabled, refine the answer
    if use_meta_cognitive and confidence < confidence_threshold:
        # Dynamic scaling of iterations based on confidence level
        # Lower confidence -> more iterations needed (up to max)
        if confidence < 30:
            dynamic_iterations = meta_max_iterations
        elif confidence < 45:
            dynamic_iterations = max(2, meta_max_iterations - 1)
        else:
            dynamic_iterations = max(1, meta_max_iterations - 2)
        
        print(f"\n=== Low Confidence Detected ===")
        print(f"Confidence level: {confidence:.1f}% (below threshold of {confidence_threshold}%)")
        print(f"Applying meta-cognitive feedback with {dynamic_iterations} iterations...")
        
        # Import here to avoid circular imports
        from tandem_core import meta_evaluate_and_refine_answer
        
        # Get problem type from metadata if available
        problem_category = metadata.get("problem_category", None)
        
        # Determine focus areas based on confidence level
        focused_areas = []
        if confidence < 30:
            focused_areas = ["correctness", "reasoning", "completeness"]
        elif confidence < 45:
            focused_areas = ["reasoning", "clarity"]
        
        # Apply enhanced meta-cognitive feedback
        refinement_results = meta_evaluate_and_refine_answer(
            selected_answer, 
            problem, 
            model=model, 
            max_iterations=dynamic_iterations,
            problem_type=problem_category,
            focused_areas=focused_areas
        )
        
        # Update the selected answer
        selected_answer = refinement_results["final_answer"]
        meta_cognitive_applied = True
        meta_iterations_used = refinement_results["iterations_used"]
        
        # Store refinement metadata for reporting
        meta_feedback_history = refinement_results.get("feedback_history", [])
        meta_improvement_history = refinement_results.get("improvement_history", [])
        
        # Update metadata to include refinement information
        metadata["meta_feedback_history"] = meta_feedback_history
        metadata["meta_improvement_history"] = meta_improvement_history
        metadata["issues_resolved"] = refinement_results.get("issues_resolved", 0)
    
    # Verify the selected answer
    print("\n=== Beginning Solution Verification ===")
    is_correct, feedback = verify_answer_correctness(selected_answer, problem, model)
    
    # Prepare the results
    results = {
        "solution": selected_answer,
        "metadata": metadata,
        "verification": {
            "is_correct": is_correct,
            "feedback": feedback
        },
        "meta_cognitive_applied": meta_cognitive_applied,
        "meta_iterations_used": meta_iterations_used
    }
    
    return results