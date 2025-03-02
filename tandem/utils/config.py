"""
Configuration settings for the Tandem framework.

This module contains centralized configuration settings to avoid duplication
across different components of the system.
"""

# Modelfile paths
MODELFILE_PATHS = {
    'BASE': "tandem/utils/modelfile/llama3.3-70b",  # Default base model
    'CLASSIFIER': "tandem/utils/modelfile/llama3.3-classifier",  # Classifier model
    'M1': "tandem/utils/modelfile/llama3.3-m1",  # M1-optimized model
    # To use Qwen model, change 'BASE' to "tandem/utils/modelfile/qwen2.5-72b"
}