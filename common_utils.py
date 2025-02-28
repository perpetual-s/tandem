from ollama import chat
from typing import List, Tuple, Callable, Any
import time

def read_prompt_from_file(filename: str = "prompt.txt") -> str:
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return ""

def time_function_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_model_response(prompt: str, model: str) -> Tuple[str, bool]:
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
