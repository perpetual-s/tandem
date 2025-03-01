#!/usr/bin/env python3
"""
Command-line interface for the Tandem framework.
"""

import argparse
import sys
from tandem.utils.common import read_prompt_from_file
from tandem.models.manager import prepare_model, delete_model
from tandem.core.self_consistency import self_consistency_solve

def main():
    """
    Main entry point for the Tandem CLI.
    """
    parser = argparse.ArgumentParser(
        description="Tandem - A Framework for Enhancing Local LLMs",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("prompt_file", type=str, nargs="?", default="prompt.txt",
                      help="Path to the prompt file (default: prompt.txt)")
    
    parser.add_argument("--iterations", "-i", type=int, default=3, 
                      help="Number of solutions to generate for self-consistency (default: 3)")
    
    parser.add_argument("--confidence", "-c", type=float, default=80.0,
                     help="Confidence threshold %% below which to apply meta-cognitive feedback (default: 80.0)")
    
    parser.add_argument("--meta", action="store_true", default=True,
                     help="Use meta-cognitive feedback for low confidence answers (default: True)")
    
    parser.add_argument("--no-meta", action="store_false", dest="meta",
                     help="Disable meta-cognitive feedback")
    
    parser.add_argument("--meta-iterations", "-m", type=int, default=3,
                     help="Maximum number of meta-cognitive iterations (default: 3)")
    
    parser.add_argument("--version", "-v", action="version", version="Tandem 0.1.0")
    
    args = parser.parse_args()
    
    # Read the problem from the prompt file
    problem = read_prompt_from_file(args.prompt_file)
    if not problem:
        print(f"Error: No problem found in the prompt file: {args.prompt_file}")
        sys.exit(1)
    
    print("\n=== Problem Statement ===")
    print(problem)
    
    # Prepare an optimized model for the problem
    temp_model = prepare_model(problem)
    if not temp_model:
        print("Error preparing specialized model.")
        sys.exit(1)
    
    try:
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
        
        print("\n=== Final Solution ===")
        print(results["solution"])
        
    finally:
        # Clean up: Delete the temporary model
        delete_model(temp_model)

if __name__ == "__main__":
    main()