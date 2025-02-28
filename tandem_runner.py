from tandem_core import read_prompt_from_file
import model_manager
from self_consistency import self_consistency_solve
import argparse

# Global variables for modelfile locations.
ORIGINAL_MODELFIL_PATH = "_modelfile/llama3.3-70b"
CLASSIFIER_MODELFIL_PATH = "_modelfile/llama3.3-classifier"

def main():
    # You can customize the behavior with various flags:
    # python tandem_runner.py --iterations 5 --confidence 70 --meta-iterations 4

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Project Tandem - LLM Enhancement Pipeline")
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
    
    # Step 0: Read the overall query from prompt.txt.
    query = read_prompt_from_file()
    if not query:
        print("No Query provided. Exiting.")
        return

    print("\n=== Problem Statement ===")
    print(query)

    # Step 1: Prepare a temporary model based on the query's classification.
    temp_model = model_manager.prepare_temporary_model(query)
    if not temp_model:
        print("Error preparing temporary model.")
        return
    print(f"\n=== Model Preparation ===")
    print(f"Using optimized model: {temp_model}")

    # Step 2: Use self-consistency to solve the problem
    print(f"\nUsing self-consistency approach with {args.iterations} solutions")
    
    results = self_consistency_solve(
        query, 
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
    
    final_answer = results["solution"]

    # Clean up: Delete temporary model.
    model_manager.delete_temporary_model(temp_model)
    
    print("\n=== Final Solution ===")
    print(final_answer)

if __name__ == "__main__":
    main()