"""
Tandem: A framework for enhancing local Large Language Models.

This package provides tools to improve the performance of locally-run LLMs
through problem classification, parameter optimization, self-consistency
solution generation, and meta-cognitive feedback.
"""

__version__ = "0.1.0"
__author__ = "Chaeho Shin"

from tandem.core.self_consistency import self_consistency_solve
from tandem.core.meta_cognitive import meta_evaluate_and_refine_answer
from tandem.utils.common import read_prompt_from_file, get_model_response
from tandem.models.manager import prepare_model, classify_problem