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
    parser.add_argument("--confidence", type=float, default=80.0,
                     help="Confidence threshold below which to apply meta-cognitive feedback")
    parser.add_argument("--meta", action="store_true", default=True,
                     help="Use meta-cognitive feedback for low confidence answers")
    parser.add_argument("--no-meta", action="store_false", dest="meta",
                     help="Disable meta-cognitive feedback")
    parser.add_argument("--meta-iterations", type=int, default=3,
                     help="Maximum number of meta-cognitive iterations")
    parser.add_argument("--hybridize", action="store_true", default=True,
                     help="Use solution hybridization for low confidence answers")
    parser.add_argument("--no-hybridize", action="store_false", dest="hybridize",
                     help="Disable solution hybridization")
    parser.add_argument("--file", action="store_true", 
                     help="Read query from prompt.txt instead of user input")
    args = parser.parse_args()
    
    # Step 0: Get query from user input or prompt.txt
    print_header("Project Tandem")
    colored_print("An AI-powered problem-solving framework", Colors.YELLOW)
    print("\n")
    
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
        colored_print("Type or paste your question below and press Enter.", Colors.YELLOW)
        colored_print("For single-line input, just type and press Enter.", Colors.YELLOW)
        colored_print("For multi-line input, add your problem to prompt.txt and use the --file flag.", Colors.YELLOW)
        print()
        
        # Simple approach that works reliably for single-line inputs
        print("➤ ", end="")
        query = input()
        
        # Handle quoted text (often from pasting)
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        
        if not query.strip():
            colored_print("No query provided. Exiting.", Colors.RED, bold=True)
            return

    print_header("Problem Statement")
    # Indent and style the problem statement to make it stand out
    lines = query.strip().split("\n")
    for line in lines:
        print(f"{Colors.YELLOW}  {line}{Colors.ENDC}")
    print("\n")

    # Step 1: Prepare a temporary model based on the query's classification
    print_step(2, "Creating specialized model for your problem")
    
    colored_print("Classifying problem...", Colors.YELLOW)
    temp_model, problem_category = model_manager.prepare_temporary_model(query, return_category=True)
    
    if not temp_model:
        colored_print("Error preparing temporary model.", Colors.RED, bold=True)
        return
        
    colored_print(f"\nProblem classified as: {problem_category}", Colors.GREEN)
    colored_print(f"Using optimized model: {temp_model}", Colors.GREEN)
    print("\n")

    # Step 2: Use self-consistency to solve the problem
    print_step(3, f"Solving with self-consistency ({args.iterations} iterations)")
    
    # Display configuration
    colored_print(f"✓ Generating {args.iterations} diverse solutions", Colors.GREEN)
    
    if args.meta:
        colored_print(f"✓ Meta-cognitive feedback enabled", Colors.GREEN)
        colored_print(f"  • Confidence threshold: {args.confidence}%", Colors.GREEN)
        colored_print(f"  • Max meta iterations: {args.meta_iterations}", Colors.GREEN)
    else:
        colored_print("✗ Meta-cognitive feedback disabled", Colors.YELLOW)
        
    if args.hybridize:
        colored_print(f"✓ Solution hybridization enabled", Colors.GREEN)
    else:
        colored_print("✗ Solution hybridization disabled", Colors.YELLOW)
    
    print("\n")
    
    results = self_consistency_solve(
        query, 
        model=temp_model, 
        num_answers=args.iterations,
        use_meta_cognitive=args.meta,
        confidence_threshold=args.confidence,
        meta_max_iterations=args.meta_iterations,
        problem_category=problem_category,  # Pass the category to avoid reclassification
        use_hybridization=args.hybridize
    )
    
    # Print the solution with enhanced UI
    print_header("Solution Results")
    print("\n")
    
    # Print agreement statistics in a visually appealing way
    metadata = results["metadata"]
    agreement = metadata.get('agreement_percentage', 0)
    
    # Color-coded agreement indicator
    agreement_color = Colors.GREEN if agreement >= 80 else Colors.YELLOW if agreement >= 60 else Colors.RED
    
    print("📊 Solution Agreement")
    print(f"   Agreement: ", end="")
    colored_print(f"{agreement:.1f}%", agreement_color, bold=True)
    
    # Display answer distribution with weighted scoring
    if metadata.get('answer_counts') and metadata.get('weighted_scores'):
        print("\n📊 Answer Evaluation:")
        answer_counts = metadata.get('answer_counts', {})
        weighted_scores = metadata.get('weighted_scores', {})
        max_count = max(answer_counts.values()) if answer_counts else 0
        max_weight = max(weighted_scores.values()) if weighted_scores else 0
        total_answers = sum(answer_counts.values())
        
        # Sort answers by weighted score (descending)
        sorted_by_weight = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display header
        print(f"   {'Answer':<20} {'Count':<8} {'Weight':<8} {'Quality'}")
        
        for answer, weight in sorted_by_weight:
            count = answer_counts[answer]
            
            # Calculate percentage and bar length based on weight
            bar_length = int((weight / max_weight) * 20) if max_weight > 0 else 0
            bar = "█" * bar_length
            
            # Choose color based on weight relative to max
            color = Colors.GREEN if weight == max_weight else Colors.YELLOW
            
            # Print the bar with count and weight
            answer_display = answer[:18] + '..' if len(answer) > 20 else answer
            print(f"   {answer_display:<20} {count:<8} {weight:<8.1f}", end="")
            colored_print(f"{bar}", color)
            
        # Explain the quality scoring
        print("\n   Quality factors considered:")
        print("   • Verification steps (+0.5)")
        print("   • Structured format (+0.5)")
        print("   • Step-by-step reasoning (+0.3)")
        print("   • Detailed work (+0.4)")
        print("   • Appropriate length (+0.3)")
    
    # Verification results
    print("\n\n✅ Verification")
    is_correct = results['verification']['is_correct']
    verification_color = Colors.GREEN if is_correct else Colors.YELLOW
    verification_symbol = "✓" if is_correct else "!"
    
    print(f"   Correct: ", end="")
    colored_print(f"{verification_symbol} {is_correct}", verification_color, bold=True)
    print(f"   Feedback: {results['verification']['feedback']}\n")
    
    # Show solution enhancement details
    print_header("Solution Enhancement")
    
    # Show hybridization information if applied
    if results.get("hybridization_applied", False):
        print("\n🧠 Solution Hybridization")
        colored_print("✓ Applied: Combined multiple solution approaches", Colors.GREEN, bold=True)
        if metadata.get("hybridization_note"):
            print(f"📝 {metadata['hybridization_note']}")
        print("🧩 Created an optimal hybrid answer using the best reasoning from multiple solutions")
    
    # Meta-cognitive information with better visual indicators
    if results["meta_cognitive_applied"]:
        print("\n🔍 Meta-Cognitive Refinement")
        
        # Display actual iterations used
        print(f"🔄 Iterations used: {results['meta_iterations_used']}")
        
        # Display how many iterations were needed based on confidence level
        agreement = metadata.get('agreement_percentage', 0)
        if agreement < 40:
            print(f"🔄 Required iterations for confidence level {agreement:.1f}%: 5")
        elif agreement < 80:
            print(f"🔄 Required iterations for confidence level {agreement:.1f}%: 3")
        else:
            print(f"🔄 Required iterations for confidence level {agreement:.1f}%: {args.meta_iterations}")
            
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
            
        # Display final verification if available
        if metadata.get("final_verification"):
            verification_text = metadata["final_verification"]
            shortened = f"{verification_text[:150]}..." if len(verification_text) > 150 else verification_text
            print(f"\n✅ Final verification result:")
            print(f"   {shortened}")
    
    final_answer = results["solution"]

    # Clean up
    print("\n🧹 Cleaning up...")
    colored_print("Removing temporary model...", Colors.YELLOW)
    model_manager.delete_temporary_model(temp_model)
    
    # Final solution display
    print_header("Final Solution")
    print("\n")
    
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
    print("\n")
    colored_print("✨ Solution process complete!", Colors.GREEN, bold=True)
    print("\n")

if __name__ == "__main__":
    main()