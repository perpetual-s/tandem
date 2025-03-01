"""
Model management utilities for the Tandem framework.

This module handles model creation, parameter optimization, and problem classification
using the Ollama API.
"""

import os
import re
import time
from typing import Optional
from pathlib import Path
from tandem.utils.common import get_model_response, resolve_path

# Default paths for modelfiles
DEFAULT_BASE_MODEL_PATH = "_modelfile/llama3.3-70b"
DEFAULT_CLASSIFIER_PATH = "_modelfile/llama3.3-classifier"

def classify_problem(problem: str, classifier_modelfile_path: str = DEFAULT_CLASSIFIER_PATH) -> str:
    """
    Classifies a problem into a category (math, coding, writing, etc.).
    
    Args:
        problem: The problem text to classify
        classifier_modelfile_path: Path to the classifier modelfile
        
    Returns:
        The problem category as a string
    """
    temp_classifier_model_name = "temp_classifier_model"
    
    # Read the classifier modelfile content
    try:
        with open(classifier_modelfile_path, 'r') as file:
            classifier_modelfile_content = file.read().strip()
    except FileNotFoundError:
        print(f"Classifier modelfile '{classifier_modelfile_path}' not found. Defaulting classification to 'General'.")
        return "General"
    
    # Write classifier modelfile content to a temporary file
    temp_classifier_file = "temp_classifier_modelfile.txt"
    try:
        with open(temp_classifier_file, "w") as f:
            f.write(classifier_modelfile_content)
    except Exception as e:
        print(f"Error writing temporary classifier modelfile: {e}")
        return "General"
    
    # Create the temporary classifier model using CLI command
    create_command = f"ollama create {temp_classifier_model_name} -f {temp_classifier_file}"
    print(f"\n=== Creating Classifier Model ===")
    print(f"Executing: {create_command}")
    os.system(create_command)
    print(f"Classifier model created successfully.")
    
    # Use the classifier to get the category
    response, success = get_model_response(problem, model=temp_classifier_model_name)
    
    # Delete the temporary classifier model
    delete_command = f"ollama rm {temp_classifier_model_name}"
    print(f"\n=== Cleaning Up Resources ===")
    print(f"Executing: {delete_command}")
    os.system(delete_command)
    print(f"Temporary classifier model deleted successfully.")
    
    # Remove the temporary classifier modelfile
    try:
        os.remove(temp_classifier_file)
    except Exception as e:
        print(f"Warning: unable to delete temporary file {temp_classifier_file}: {e}")
    
    # Extract the category using regex
    match = re.search(r"classified as\s+([\w\s&]+)[\.\n]", response, re.IGNORECASE)
    if match:
        category = match.group(1).strip()
        return category
    else:
        print("Could not parse classification output. Defaulting to 'General'.")
        return "General"

def get_optimal_parameters(problem_type: str) -> str:
    """
    Returns optimized parameter settings for the modelfile based on the problem category.
    
    Args:
        problem_type: The type of problem (math, coding, writing, etc.)
        
    Returns:
        String containing the Ollama parameter settings
    """
    base_params = (
        "PARAMETER num_ctx 2048\n"  # Default context window size
    )
    
    category = problem_type.lower()
    if category == "math":
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
    # Add other categories here...
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

def create_modelfile(original_modelfile_path: str, additional_params: str) -> str:
    """
    Creates a modelfile content by appending additional parameters to the original modelfile.
    
    Args:
        original_modelfile_path: Path to the base modelfile
        additional_params: Parameters to append to the modelfile
        
    Returns:
        Complete modelfile content as a string
    """
    try:
        with open(original_modelfile_path, 'r') as file:
            original_contents = file.read().strip()
    except FileNotFoundError:
        print(f"Warning: {original_modelfile_path} not found. Using default base modelfile content.")
        original_contents = ""
    
    # Combine the original contents with the additional parameters
    temporary_modelfile = original_contents + "\n" + additional_params
    return temporary_modelfile

def create_model(model_name: str, modelfile_content: str) -> bool:
    """
    Creates an Ollama model using the provided modelfile content.
    
    Args:
        model_name: Name to give the model
        modelfile_content: Content of the modelfile
        
    Returns:
        Boolean indicating success
    """
    temp_modelfile = "temp_modelfile.txt"
    try:
        with open(temp_modelfile, "w") as f:
            f.write(modelfile_content)
    except Exception as e:
        print(f"Error writing temporary modelfile: {e}")
        return False

    command = f"ollama create {model_name} -f {temp_modelfile}"
    print(f"\n=== Creating Specialized Model ===")
    print(f"Executing: {command}")
    os.system(command)
    print(f"Model '{model_name}' created successfully with optimized parameters.")
    return True

def delete_model(model_name: str) -> bool:
    """
    Deletes an Ollama model.
    
    Args:
        model_name: Name of the model to delete
        
    Returns:
        Boolean indicating success
    """
    command = f"ollama rm {model_name}"
    print(f"\n=== Cleaning Up Resources ===")
    print(f"Executing: {command}")
    os.system(command)
    print(f"Model '{model_name}' deleted successfully.")
    return True

def prepare_model(problem: str, 
                  original_modelfile_path: str = DEFAULT_BASE_MODEL_PATH,
                  classifier_modelfile_path: str = DEFAULT_CLASSIFIER_PATH) -> Optional[str]:
    """
    Prepares a model optimized for the given problem by:
    1. Classifying the problem
    2. Generating optimal parameters
    3. Creating a specialized model
    
    Args:
        problem: The problem text
        original_modelfile_path: Path to the base modelfile
        classifier_modelfile_path: Path to the classifier modelfile
        
    Returns:
        The name of the created model, or None if creation failed
    """
    # Classify the problem
    category = classify_problem(problem, classifier_modelfile_path=classifier_modelfile_path)
    print(f"\n=== Problem Classification ===")
    print(f"Category: {category}")
    
    # Get optimized parameters for this problem type
    additional_params = get_optimal_parameters(category)
    
    # Create the modelfile
    temporary_modelfile = create_modelfile(original_modelfile_path, additional_params)
    
    # Create a model name based on the category
    model_name = f"temp_{category.lower().replace(' ', '_')}_model"
    
    # Create the model
    if create_model(model_name, temporary_modelfile):
        return model_name
    else:
        return None