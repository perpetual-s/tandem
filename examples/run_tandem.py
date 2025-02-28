#!/usr/bin/env python3
"""
Example script for using the Tandem framework.

This script demonstrates how to use Tandem to solve problems
with self-consistency and meta-cognitive feedback.
"""

import argparse
import sys
import os

# Add the parent directory to the path so we can import the tandem package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tandem.utils.common import read_prompt_from_file
from tandem.models.manager import prepare_model, delete_model
from tandem.core.self_consistency import self_consistency_solve

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tandem - A Framework for Enhancing Local LLMs")
    parser.add_argument("--prompt", type=str, default="prompt.txt",
                      help="Path to the prompt file (default: prompt.txt)")
    parser.add_argument("--iterations", type=int, default=3, 
                      help="Number of solutions to generate for self-consistency")
    parser.add_argument("--confidence", type=float, default=60.0,
                     help="Confidence threshold (%) below which to apply meta-cognitive feedback")
    parser.add_argument("--meta", action="store_true", default=True,
                     help="Use meta-cognitive feedback for low confidence answers")
    parser.add_argument("--no-meta", action="store_false", dest="meta",
                     help="Disable meta-cognitive feedback")
    parser.add_argument("--meta-iterations", type=int, default=3,
                     help="Maximum number of meta-cognitive iterations")
    args = parser.parse_args()
    
    # Read the problem from the prompt file
    problem = read_prompt_from_file(args.prompt)
    if not problem:
        print("Error: No problem found in the prompt file.")
        return
    
    print("\n=== Problem Statement ===")
    print(problem)
    
    # Prepare an optimized model for the problem
    temp_model = prepare_model(problem)
    if not temp_model:
        print("Error preparing specialized model.")
        return
    
    print(f"\n=== Model Preparation ===")
    print(f"Using optimized model: {temp_model}")
    
    # Solve the problem using self-consistency
    results = self_consistency_solve(
        problem, 
        model=temp_model, 
        num_answers=args.iterations,
        use_meta_cognitive=args.meta,
        confidence_threshold=args.confidence,
        meta_max_iterations=args.meta_iterations
    )
    
    # Print the solution and metadata
    print("\n=== Self-Consistency Solution ===")
    print(results["solution"])
    
    print("\n=== Verification Results ===")
    print(f"Correct: {results['verification']['is_correct']}")
    print(f"Feedback: {results['verification']['feedback']}")
    
    # Print agreement statistics
    metadata = results["metadata"]
    print("\n=== Solution Agreement ===")
    print(f"Agreement Percentage: {metadata.get('agreement_percentage', 0):.1f}%")
    print(f"Answer Distribution: {metadata.get('answer_counts', {})}")
    
    # Print meta-cognitive information
    if results["meta_cognitive_applied"]:
        print(f"\n=== Meta-Cognitive Refinement Applied ===")
        print(f"Iterations used: {results['meta_iterations_used']}")
        print(f"Confidence was below threshold of {args.confidence}%")
        print(f"Problem category: {metadata.get('problem_category', 'Not classified')}")
        
        # Print improvement history if available
        if metadata.get("meta_improvement_history"):
            print("\nImprovements made:")
            for i, improvement in enumerate(metadata["meta_improvement_history"]):
                print(f"  {i+1}. {improvement[:100]}..." if len(improvement) > 100 else f"  {i+1}. {improvement}")
                
        # Print issues resolved
        if "issues_resolved" in metadata:
            print(f"\nTotal issues resolved: {metadata['issues_resolved']}")
    
    # Clean up: Delete the temporary model
    delete_model(temp_model)
    
    print("\n=== Final Solution ===")
    print(results["solution"])

if __name__ == "__main__":
    main()