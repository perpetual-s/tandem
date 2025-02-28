import argparse
import sys

# Import utility functions
from common_utils import (
    read_prompt_from_file, 
    colored_print, 
    print_header, 
    print_step, 
    Colors
)

# Global variables for modelfile locations.
ORIGINAL_MODELFIL_PATH = "_modelfile/llama3.3-70b"
CLASSIFIER_MODELFIL_PATH = "_modelfile/llama3.3-classifier"

# Import these after defining the paths to avoid circular imports
import model_manager
from self_consistency import self_consistency_solve

def main():
    # You can customize the behavior with various flags:
    # python tandem_runner.py --iterations 5 --confidence 70 --meta-iterations 4

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Project Tandem - LLM Enhancement Pipeline")
    parser.add_argument("--iterations", type=int, default=3, 
                      help="Number of solutions to generate for self-consistency")
    parser.add_argument("--confidence", type=float, default=60.0,
                     help="Confidence threshold below which to apply meta-cognitive feedback")
    parser.add_argument("--meta", action="store_true", default=True,
                     help="Use meta-cognitive feedback for low confidence answers")
    parser.add_argument("--no-meta", action="store_false", dest="meta",
                     help="Disable meta-cognitive feedback")
    parser.add_argument("--meta-iterations", type=int, default=3,
                     help="Maximum number of meta-cognitive iterations")
    parser.add_argument("--file", action="store_true", 
                     help="Read query from prompt.txt instead of user input")
    args = parser.parse_args()
    
    # Step 0: Get query from user input or prompt.txt
    print_header("Project Tandem")
    colored_print("An AI-powered problem-solving framework", Colors.YELLOW)
    print()
    
    print_step(1, "Getting problem statement")
    
    if args.file:
        # Read from prompt.txt if --file flag is provided
        colored_print("Reading from prompt.txt...", Colors.YELLOW)
        query = read_prompt_from_file()
            
        if not query:
            colored_print("No query found in prompt.txt. Exiting.", Colors.RED, bold=True)
            return
    else:
        # Get query from user input with a more attractive prompt
        colored_print("\nEnter Your Problem", Colors.BLUE, bold=True)
        colored_print("Type your question or problem below and press Enter when finished:", Colors.YELLOW)
        print(f"{Colors.GREEN}> {Colors.ENDC}", end="")
        query = input()
        
        if not query.strip():
            colored_print("No query provided. Exiting.", Colors.RED, bold=True)
            return

    print_header("Problem Statement")
    # Indent and style the problem statement to make it stand out
    lines = query.strip().split("\n")
    for line in lines:
        print(f"{Colors.YELLOW}  {line}{Colors.ENDC}")

    # Step 1: Prepare a temporary model based on the query's classification
    print_step(2, "Creating specialized model for your problem")
    
    colored_print("Classifying problem...", Colors.YELLOW)
    temp_model, problem_category = model_manager.prepare_temporary_model(query, return_category=True)
    
    if not temp_model:
        colored_print("Error preparing temporary model.", Colors.RED, bold=True)
        return
        
    colored_print(f"Problem classified as: {problem_category}", Colors.GREEN)
    colored_print(f"Using optimized model: {temp_model}", Colors.GREEN)

    # Step 2: Use self-consistency to solve the problem
    print_step(3, f"Solving with self-consistency ({args.iterations} iterations)")
    
    # Display configuration
    if args.meta:
        colored_print(f"✓ Meta-cognitive feedback enabled", Colors.GREEN)
        colored_print(f"✓ Confidence threshold: {args.confidence}%", Colors.GREEN)
        colored_print(f"✓ Max meta iterations: {args.meta_iterations}", Colors.GREEN)
    else:
        colored_print("✗ Meta-cognitive feedback disabled", Colors.YELLOW)
    
    results = self_consistency_solve(
        query, 
        model=temp_model, 
        num_answers=args.iterations,
        use_meta_cognitive=args.meta,
        confidence_threshold=args.confidence,
        meta_max_iterations=args.meta_iterations,
        problem_category=problem_category  # Pass the category to avoid reclassification
    )
    
    # Print the solution with enhanced UI
    print_header("Solution Results")
    print()
    
    # Print agreement statistics in a visually appealing way
    metadata = results["metadata"]
    agreement = metadata.get('agreement_percentage', 0)
    
    # Color-coded agreement indicator
    agreement_color = Colors.GREEN if agreement >= 80 else Colors.YELLOW if agreement >= 60 else Colors.RED
    
    print("📊 Solution Agreement")
    print(f"   Agreement: ", end="")
    colored_print(f"{agreement:.1f}%", agreement_color, bold=True)
    
    # Display answer distribution as a horizontal bar chart
    if metadata.get('answer_counts'):
        print("\n📈 Answer Distribution:")
        answer_counts = metadata.get('answer_counts', {})
        max_count = max(answer_counts.values())
        total_answers = sum(answer_counts.values())
        
        # Sort answers by count (descending)
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        
        for answer, count in sorted_answers:
            # Calculate percentage and bar length
            percentage = (count / total_answers) * 100
            bar_length = int((count / max_count) * 20)  # Scale to max 20 chars
            bar = "█" * bar_length
            
            # Choose color based on count relative to max
            color = Colors.GREEN if count == max_count else Colors.YELLOW
            
            # Print the bar with count and percentage
            print(f"   {answer}: ", end="")
            colored_print(f"{bar} {count} ({percentage:.1f}%)", color)
    
    # Verification results
    print("\n✅ Verification")
    is_correct = results['verification']['is_correct']
    verification_color = Colors.GREEN if is_correct else Colors.YELLOW
    verification_symbol = "✓" if is_correct else "!"
    
    print(f"   Correct: ", end="")
    colored_print(f"{verification_symbol} {is_correct}", verification_color, bold=True)
    print(f"   Feedback: {results['verification']['feedback']}")
    
    # Meta-cognitive information with better visual indicators
    if results["meta_cognitive_applied"]:
        print_header("Meta-Cognitive Refinement")
        
        print(f"🔄 Iterations used: {results['meta_iterations_used']}")
        print(f"🎯 Confidence threshold: {args.confidence}%")
        print(f"🧩 Problem category: {metadata.get('problem_category', 'Not classified')}")
        
        # Print improvement history with better formatting
        if metadata.get("meta_improvement_history"):
            print("\n📝 Improvements made:")
            for i, improvement in enumerate(metadata["meta_improvement_history"]):
                shortened = f"{improvement[:95]}..." if len(improvement) > 95 else improvement
                print(f"   {i+1}. {shortened}")
                
        # Print issues resolved
        if "issues_resolved" in metadata:
            print(f"\n🛠️ Total issues resolved: {metadata['issues_resolved']}")
    
    final_answer = results["solution"]

    # Clean up
    print("\n🧹 Cleaning up...")
    colored_print("Removing temporary model...", Colors.YELLOW)
    model_manager.delete_temporary_model(temp_model)
    
    # Final solution display
    print_header("Final Solution")
    
    # Add a styled box around the solution for emphasis
    print("┌" + "─" * 78 + "┐")
    solution_lines = final_answer.strip().split("\n")
    for line in solution_lines:
        # Wrap long lines
        while len(line) > 76:
            print(f"│ {line[:76]} │")
            line = line[76:]
        print(f"│ {line.ljust(76)} │")
    print("└" + "─" * 78 + "┘")
    
    # Final message
    print()
    colored_print("✨ Solution process complete!", Colors.GREEN, bold=True)

if __name__ == "__main__":
    main()