"""
Common utility functions for the Tandem framework.

This module provides helper functions for file I/O, model interaction,
and time measurement.
"""

from ollama import chat
from typing import List, Tuple, Callable, Any
import time
import os
from pathlib import Path

def read_prompt_from_file(filename: str = "prompt.txt") -> str:
    """
    Read and return the contents of a text file.
    
    Args:
        filename: Path to the text file to read
        
    Returns:
        The contents of the file as a string, or empty string if file not found
    """
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return ""

def time_function_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """
    Measure the execution time of a function call.
    
    Args:
        func: The function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        A tuple of (function_result, execution_time_in_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_model_response(prompt: str, model: str) -> Tuple[str, bool]:
    """
    Get a response from an LLM model via the Ollama API.
    
    Args:
        prompt: The input prompt to send to the model
        model: The name of the model to use
        
    Returns:
        A tuple of (model_response, success_flag)
    """
    try:
        response_chunks: List[str] = []
        model_display_name = model.split(':')[0].upper()
        print(f"\n{model_display_name} Responding:")
        print("-" * 40)
        
        def chat_with_model():
            stream = chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            for chunk in stream:
                content = chunk['message']['content']
                response_chunks.append(content)
                print(content, end='', flush=True)
            return ''.join(response_chunks)
        
        full_response, elapsed_time = time_function_call(chat_with_model)
        print("\n" + "-" * 40)
        print(f"{model_display_name} took {elapsed_time:.2f} seconds to respond.\n")
        return full_response, True
    except Exception as e:
        print(f"Error occurred with {model}: {e}")
        return "", False

def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Path object pointing to the project root
    """
    # This assumes the module is in tandem/utils/common.py
    return Path(__file__).parent.parent.parent

def resolve_path(relative_path: str) -> str:
    """
    Resolve a path relative to the project root.
    
    Args:
        relative_path: Path relative to project root
        
    Returns:
        Absolute path as string
    """
    root = get_project_root()
    return str(root / relative_path)