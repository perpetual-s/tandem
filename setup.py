#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tandem",
    version="0.1.0",
    author="Chaeho Shin",
    author_email="cogh0972@gmail.com",
    description="A framework for enhancing local Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/perpetual-s/tandem",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ollama",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "tandem=tandem.cli:main",
        ],
    },
)