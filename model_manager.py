import os
import re
from common_utils import get_model_response, time_function_call 

# Define paths here to avoid circular imports
ORIGINAL_MODELFIL_PATH = "_modelfile/llama3.3-70b"
CLASSIFIER_MODELFIL_PATH = "_modelfile/llama3.3-classifier"

def classify_problem(problem: str, classifier_modelfile_path: str = CLASSIFIER_MODELFIL_PATH) -> str:
    """
    Creates a temporary classifier model using the provided classifier modelfile,
    uses it to classify the problem, and then deletes the classifier.
    
    The classifier is expected to respond with:
      "This query has been classified as [CATEGORY]."
      
    Returns the extracted category or defaults to 'General' if something goes wrong.
    """
    temp_classifier_model_name = "temp_classifier_model"
    
    # Read the classifier modelfile content.
    try:
        with open(classifier_modelfile_path, 'r') as file:
            classifier_modelfile_content = file.read().strip()
    except FileNotFoundError:
        print(f"Classifier modelfile '{classifier_modelfile_path}' not found. Defaulting classification to 'General'.")
        return "General"
    
    # Write classifier modelfile content to a temporary file.
    temp_classifier_file = "temp_classifier_modelfile.txt"
    try:
        with open(temp_classifier_file, "w") as f:
            f.write(classifier_modelfile_content)
    except Exception as e:
        print(f"Error writing temporary classifier modelfile: {e}")
        return "General"
    
    # Create the temporary classifier model using CLI command.
    create_command = f"ollama create {temp_classifier_model_name} -f {temp_classifier_file}"
    print(f"\nExecuting command: {create_command}")
    os.system(create_command)
    print(f"Temporary classifier model '{temp_classifier_model_name}' created successfully.")
    
    # Use the classifier to get the category.
    response, success = get_model_response(problem, model=temp_classifier_model_name)
    
    # Delete the temporary classifier model.
    delete_command = f"ollama rm {temp_classifier_model_name}"
    print(f"Executing command: {delete_command}")
    os.system(delete_command)
    print(f"Temporary classifier model '{temp_classifier_model_name}' deleted successfully.")
    
    # Remove the temporary classifier modelfile.
    try:
        os.remove(temp_classifier_file)
    except Exception as e:
        print(f"Warning: unable to delete temporary file {temp_classifier_file}: {e}")
    
    # Extract the category using regex.
    match = re.search(r"classified as\s+([\w\s&]+)[\.\n]", response, re.IGNORECASE)
    if match:
        category = match.group(1).strip()
        return category
    else:
        print("Could not parse classification output. Defaulting to 'General'.")
        return "General"

def get_additional_parameters(problem_type: str, stage: str = None) -> str:
    """
    Returns optimized parameter settings for the modelfile based on the problem category
    and processing stage.
    
    Parameters:
    - problem_type: The category of problem (math, coding, writing, etc.)
    - stage: The processing stage (exploration, calculation, verification, or None for default)
    
    Ollama supported parameters:
    - temperature: Controls randomness (lower = more deterministic) [Default: 0.8]
    - top_p: Nucleus sampling threshold (lower = more focused) [Default: 0.9]
    - top_k: Limits vocabulary to top K tokens (lower = more conservative) [Default: 40]
    - num_ctx: Context window size in tokens [Default: 2048]
    - seed: Random seed (fixed seed = deterministic output, 0 = random) [Default: 0]
    - repeat_penalty: Penalty for repetition (higher = less repetition) [Default: 1.1]
    - repeat_last_n: How far back to look for repetitions [Default: 64]
    - min_p: Minimum probability threshold [Default: 0.05]
    - mirostat: Sampling algorithm (0=disabled, 1=Mirostat, 2=Mirostat 2.0) [Default: 0]
    """
    base_params = (
        "PARAMETER num_ctx 3072\n"  # Increased context window for all categories
    )
    
    category = problem_type.lower()
    if category == "math":
        # Progressive temperature control for math problems
        if stage == "exploration":
            return (
                "# Math exploration stage - more creative for initial reasoning:\n"
                f"{base_params}"
                "PARAMETER temperature 0.7\n"      # Higher temp for exploring approaches
                "PARAMETER top_p 0.9\n"            # Wider sampling for diverse ideas
                "PARAMETER top_k 40\n"             # Standard token limit
                "PARAMETER seed 0\n"               # Random seed for diversity
                "PARAMETER repeat_penalty 1.1\n"   # Standard repetition penalty
                "PARAMETER min_p 0.05\n"           # Standard minimum probability
                "PARAMETER repeat_last_n 64\n"     # Standard repetition window
            )
        elif stage == "calculation":
            return (
                "# Math calculation stage - balanced approach:\n"
                f"{base_params}"
                "PARAMETER temperature 0.3\n"      # Moderate temp for calculations
                "PARAMETER top_p 0.5\n"            # More focused sampling
                "PARAMETER top_k 30\n"             # Limited token set
                "PARAMETER seed 42\n"              # Fixed seed for reproducibility
                "PARAMETER repeat_penalty 1.2\n"   # Moderate repetition penalty
                "PARAMETER min_p 0.08\n"           # Slightly higher minimum probability
                "PARAMETER repeat_last_n 96\n"     # Larger repetition window
            )
        elif stage == "verification":
            return (
                "# Math verification stage - high precision for checking:\n"
                f"{base_params}"
                "PARAMETER temperature 0.1\n"      # Very deterministic for verification
                "PARAMETER top_p 0.2\n"            # Highly focused sampling
                "PARAMETER top_k 20\n"             # Very limited token set
                "PARAMETER seed 42\n"              # Fixed seed for reproducibility
                "PARAMETER repeat_penalty 1.3\n"   # Strong repetition penalty
                "PARAMETER min_p 0.1\n"            # Higher minimum probability threshold
                "PARAMETER repeat_last_n 128\n"    # Large repetition window
            )
        else:
            # Default math parameters if no stage specified
            return (
                "# Optimized parameters for math problems - high precision, deterministic:\n"
                f"{base_params}"
                "PARAMETER temperature 0.1\n"      # More deterministic for math
                "PARAMETER top_p 0.2\n"            # Very focused sampling
                "PARAMETER top_k 30\n"             # Limited token set for precision
                "PARAMETER seed 42\n"              # Fixed seed for reproducibility
                "PARAMETER repeat_penalty 1.3\n"   # Stronger penalty for repetition
                "PARAMETER min_p 0.1\n"            # Filter out unlikely tokens
                "PARAMETER repeat_last_n 128\n"    # Look back further to avoid repetition
            )
    elif category == "coding":
        return (
            "# Optimized parameters for coding tasks - balanced precision and creativity:\n"
            f"{base_params}"
            "PARAMETER temperature 0.2\n"      # Slightly more deterministic
            "PARAMETER top_p 0.4\n"            # Focused but allows some variation
            "PARAMETER top_k 40\n"             # Moderate token selection
            "PARAMETER seed 42\n"              # Fixed seed for reproducible solutions
            "PARAMETER repeat_penalty 1.2\n"   # Prevent code repetition
            "PARAMETER min_p 0.05\n"           # Default filtering
            "PARAMETER repeat_last_n 128\n"    # Extended lookback for code blocks
        )
    elif category == "writing":
        return (
            "# Optimized parameters for creative writing - high diversity and fluency:\n"
            f"{base_params}"
            "PARAMETER temperature 0.8\n"      # High creativity
            "PARAMETER top_p 0.95\n"           # Wide sampling
            "PARAMETER top_k 60\n"             # Many token options
            "PARAMETER seed 0\n"               # Random seed for variety
            "PARAMETER repeat_penalty 0.9\n"   # Lower repetition penalty for natural flow
            "PARAMETER min_p 0.02\n"           # Allow more diverse tokens
            "PARAMETER repeat_last_n 64\n"     # Standard repetition window
        )
    elif category == "history":
        return (
            "# Optimized parameters for historical analysis - balanced approach:\n"
            f"{base_params}"
            "PARAMETER temperature 0.6\n"      # Moderate creativity
            "PARAMETER top_p 0.8\n"            # Balanced sampling
            "PARAMETER top_k 45\n"             # Moderate token selection
            "PARAMETER seed 0\n"               # Random seed
            "PARAMETER repeat_penalty 1.1\n"   # Moderate repetition control
            "PARAMETER min_p 0.05\n"           # Standard filtering
            "PARAMETER repeat_last_n 100\n"    # Extended for historical narratives
        )
    elif category == "science":
        return (
            "# Optimized parameters for scientific queries - precision with depth:\n"
            f"{base_params}"
            "PARAMETER temperature 0.3\n"      # More deterministic for accuracy
            "PARAMETER top_p 0.5\n"            # Focused sampling
            "PARAMETER top_k 40\n"             # Moderate token restriction
            "PARAMETER seed 0\n"               # Random seed
            "PARAMETER repeat_penalty 1.1\n"   # Prevent concept repetition
            "PARAMETER min_p 0.05\n"           # Standard filtering
            "PARAMETER repeat_last_n 100\n"    # Extended for scientific explanations
        )
    elif category == "puzzles":
        return (
            "# Optimized parameters for puzzles - highly deterministic reasoning:\n"
            f"{base_params}"
            "PARAMETER temperature 0.1\n"      # Very deterministic
            "PARAMETER top_p 0.3\n"            # Focused sampling
            "PARAMETER top_k 35\n"             # Limited token set
            "PARAMETER seed 42\n"              # Fixed seed for reproducibility
            "PARAMETER repeat_penalty 1.3\n"   # Prevent loop thinking
            "PARAMETER min_p 0.05\n"           # Standard filtering
            "PARAMETER repeat_last_n 128\n"    # Expanded window to avoid circular reasoning
        )
    elif category == "philosophy":
        return (
            "# Optimized parameters for philosophical queries - balanced depth and creativity:\n"
            f"{base_params}"
            "PARAMETER temperature 0.7\n"      # Creative but controlled
            "PARAMETER top_p 0.85\n"           # Relatively wide sampling
            "PARAMETER top_k 50\n"             # Diverse token selection
            "PARAMETER seed 0\n"               # Random seed
            "PARAMETER repeat_penalty 1.0\n"   # Standard repetition control
            "PARAMETER min_p 0.03\n"           # Allow diverse philosophical concepts
            "PARAMETER repeat_last_n 100\n"    # Extended for philosophical arguments
        )
    elif category == "trivia":
        return (
            "# Optimized parameters for trivia questions - factual accuracy focus:\n"
            f"{base_params}"
            "PARAMETER temperature 0.3\n"      # Lower randomness for factual accuracy
            "PARAMETER top_p 0.6\n"            # Moderate sampling focus
            "PARAMETER top_k 40\n"             # Moderate token restriction
            "PARAMETER seed 0\n"               # Random seed
            "PARAMETER repeat_penalty 1.1\n"   # Moderate repetition control
            "PARAMETER min_p 0.05\n"           # Standard filtering
        )
    elif category == "business & finance":
        return (
            "# Optimized parameters for business & finance queries - precision with clarity:\n"
            f"{base_params}"
            "PARAMETER temperature 0.3\n"      # Lower randomness for accuracy
            "PARAMETER top_p 0.5\n"            # Focused sampling
            "PARAMETER top_k 40\n"             # Moderate token restriction
            "PARAMETER seed 0\n"               # Random seed
            "PARAMETER repeat_penalty 1.1\n"   # Prevent repetitive explanations
            "PARAMETER min_p 0.05\n"           # Standard filtering
        )
    elif category == "self_consistency_solver":
        return (
            "# Special parameters optimized for generating diverse solutions in self-consistency approach:\n"
            f"{base_params}"
            "PARAMETER temperature 0.5\n"      # Medium temperature to enable diverse answers
            "PARAMETER top_p 0.8\n"            # Wider sampling for varied approaches
            "PARAMETER top_k 50\n"             # Allow more token options
            "PARAMETER seed 0\n"               # Random seed for variation
            "PARAMETER repeat_penalty 1.1\n"   # Moderate repetition control
            "PARAMETER min_p 0.03\n"           # Allow somewhat diverse tokens
            "PARAMETER repeat_last_n 100\n"    # Extended window for complex problems
        )
    elif category == "verification":
        return (
            "# Special parameters optimized for verification and evaluation steps:\n"
            f"{base_params}"
            "PARAMETER temperature 0.1\n"      # Very deterministic for critical analysis
            "PARAMETER top_p 0.3\n"            # Highly focused sampling
            "PARAMETER top_k 30\n"             # Limited token set for precision
            "PARAMETER seed 42\n"              # Fixed seed for consistent evaluation
            "PARAMETER repeat_penalty 1.2\n"   # Prevent repetitive critique
            "PARAMETER min_p 0.1\n"            # Filter unlikely tokens for precision
            "PARAMETER repeat_last_n 128\n"    # Extended window for thorough verification
        )
    else:
        return (
            "# Balanced parameters for general queries:\n"
            f"{base_params}"
            "PARAMETER temperature 0.5\n"
            "PARAMETER top_p 0.7\n"
            "PARAMETER top_k 45\n"
            "PARAMETER seed 0\n"
            "PARAMETER repeat_penalty 1.1\n"
            "PARAMETER min_p 0.05\n"
        )

def create_temporary_modelfile(original_modelfile_path: str = ORIGINAL_MODELFIL_PATH, additional_params: str = "") -> str:
    """
    Creates a temporary modelfile content by appending additional parameters
    to the original modelfile.
    """
    try:
        with open(original_modelfile_path, 'r') as file:
            original_contents = file.read().strip()
    except FileNotFoundError:
        print(f"Warning: {original_modelfile_path} not found. Using default base modelfile content.")
        original_contents = ""
    temporary_modelfile = original_contents + "\n" + additional_params
    return temporary_modelfile

def create_temporary_model(model_name: str, modelfile_content: str) -> bool:
    """
    Creates a temporary Ollama model using the provided modelfile content via the CLI.
    """
    temp_modelfile = "temp_modelfile.txt"
    try:
        with open(temp_modelfile, "w") as f:
            f.write(modelfile_content)
    except Exception as e:
        print(f"Error writing temporary modelfile: {e}")
        return False

    command = f"ollama create {model_name} -f {temp_modelfile}"
    print(f"\n\n=== Creating Specialized Model ===")
    print(f"Executing: {command}\n")
    os.system(command)
    print(f"\nModel '{model_name}' created successfully with optimized parameters.\n")
    return True

def delete_temporary_model(model_name: str) -> bool:
    """
    Deletes the temporary Ollama model using the CLI.
    """
    command = f"ollama rm {model_name}"
    print(f"\n\n=== Cleaning Up Resources ===")
    print(f"Executing: {command}\n")
    os.system(command)
    print(f"\nModel '{model_name}' deleted successfully.\n")
    return True

def prepare_temporary_model(problem: str, 
                              original_modelfile_path: str = ORIGINAL_MODELFIL_PATH,
                              classifier_modelfile_path: str = CLASSIFIER_MODELFIL_PATH,
                              return_category: bool = False,
                              stage: str = None):
    """
    Prepares a temporary model for the given problem by:
      1. Creating a temporary classifier to classify the problem.
      2. Generating additional parameters based on the classification and stage.
      3. Creating a temporary modelfile.
      4. Creating the temporary model using the Ollama CLI.
    
    Args:
        problem: The problem to solve
        original_modelfile_path: Path to the base modelfile
        classifier_modelfile_path: Path to the classifier modelfile
        return_category: If True, returns both model name and category
        stage: The processing stage (exploration, calculation, verification, etc.)
              This enables progressive temperature control
    
    Returns:
        If return_category is False (default): temporary model name if successful; otherwise, an empty string.
        If return_category is True: tuple of (model_name, category)
    """
    category = classify_problem(problem, classifier_modelfile_path=classifier_modelfile_path)
    print(f"\n=== Problem Classification ===")
    print(f"Category: {category}")
    
    # Get parameters specific to the category and stage (if provided)
    additional_params = get_additional_parameters(category, stage)
    temporary_modelfile = create_temporary_modelfile(original_modelfile_path, additional_params)
    temporary_model_name = f"temp_{category.lower().replace(' ', '_')}_model"
    
    if create_temporary_model(temporary_model_name, temporary_modelfile):
        if return_category:
            return temporary_model_name, category
        else:
            return temporary_model_name
    else:
        if return_category:
            return "", None
        else:
            return ""

# Example usage:
# if __name__ == "__main__":
#     sample_problem = "Solve the integral of x^2 dx over the interval [0, 1]."
#     temp_model = prepare_temporary_model(sample_problem)
#     if temp_model:
#         print(f"\nUsing temporary model: {temp_model}")
#         # ... perform operations with the temporary model ...
#         # Once done, delete the model:
#         delete_temporary_model(temp_model)
