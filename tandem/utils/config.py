"""
Configuration settings for the Tandem framework.

This module contains centralized configuration settings to avoid duplication
across different components of the system.
"""

# Modelfile paths
MODELFILE_PATHS = {
    'BASE': "_modelfile/llama3.3-70b",  # Default base model
    'CLASSIFIER': "_modelfile/llama3.3-classifier",  # Classifier model
    'M1': "_modelfile/llama3.3-m1",  # M1-optimized model
}