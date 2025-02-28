from ollama import chat
from typing import List, Tuple, Callable, Any, Optional
import time
import sys

def read_prompt_from_file(filename: str = "prompt.txt") -> str:
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return ""

# ANSI color codes for colorful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored_print(text: str, color_code: str, bold: bool = False) -> None:
    """Print colored text to the terminal."""
    bold_code = Colors.BOLD if bold else ""
    print(f"{color_code}{bold_code}{text}{Colors.ENDC}")

def print_header(text: str) -> None:
    """Print a section header with styling."""
    print("\n" + "=" * 50)
    colored_print(f" {text} ", Colors.BLUE, bold=True)
    print("=" * 50)

def print_step(step_num: int, text: str) -> None:
    """Print a step indicator with styling."""
    colored_print(f"Step {step_num}: {text}", Colors.GREEN)

def time_function_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_model_response(prompt: str, model: str) -> Tuple[str, bool]:
    """
    Get a response from the model with improved user experience.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use
        
    Returns:
        Tuple of (response_text, success)
    """
    try:
        response_chunks: List[str] = []
        model_display_name = model.split(':')[0].upper()
        
        print_header(f"{model_display_name} Responding")
        
        def chat_with_model():
            stream = chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
            )
            
            # Display a waiting message
            colored_print("Waiting for model response...", Colors.YELLOW)
            
            response_started = False
            for chunk in stream:
                content = chunk['message']['content']
                response_chunks.append(content)
                
                # Once we start getting content, switch to showing the content directly
                if content and not response_started:
                    response_started = True
                    sys.stdout.write(Colors.GREEN)  # Start green text
                
                if response_started:
                    print(content, end='', flush=True)
            
            if response_started:
                sys.stdout.write(Colors.ENDC)  # End colored text
                print()  # Add a newline at the end
                
            return ''.join(response_chunks)
        
        full_response, elapsed_time = time_function_call(chat_with_model)
        
        print("\n" + "-" * 50)
        colored_print(f"{model_display_name} response completed in {elapsed_time:.2f} seconds.", 
                      Colors.BLUE, bold=True)
        return full_response, True
    except Exception as e:
        colored_print(f"Error occurred with {model}: {e}", Colors.RED)
        return "", False
