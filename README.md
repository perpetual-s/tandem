# Project Tandem

## A Framework for Enhancing Local Large Language Models

### Performance Comparison

The following charts demonstrate the effectiveness of Project Tandem (using Llama3.3-70B) compared to the base Llama3.3-70B model on the 2024 AIME I & II mathematics competitions:

<div align="center">
<img src="assets/images/Correctness%20Comparison.png" width="350" alt="Correctness Comparison (2024 AIME I & II)"/>
<img src="assets/images/Improvement%20in%20Correctness.png" width="356" alt="Improvement in Correctness (2024 AIME I & II)"/>
</div>

Project Tandem achieved a 42.86% increase in correctness on the 2024 AIME I & II mathematics competitions.

### Introduction

Project Tandem is an advanced framework designed to significantly improve the performance of local Large Language Models (LLMs) like Llama-3.3-70B without requiring fine-tuning or model modifications. By implementing sophisticated prompting strategies, dynamic parameter optimization, and a self-consistency approach with automated verification, Tandem enables locally-run models to produce higher quality responses that rival those of much larger, cloud-based systems.

The framework addresses a critical challenge in the AI industry: how to maximize the capabilities of locally-deployed LLMs that often face performance constraints compared to their cloud-based counterparts. Tandem creates a synergistic pipeline that systematically enhances model outputs through intelligent orchestration of multiple reasoning pathways and specialized verification techniques.

### Purpose and Goals

The primary goals of Project Tandem are to:

1. **Enhance Model Performance**: Significantly improve the quality and reliability of responses from locally-run LLMs without modifying the base models.

2. **Problem-Adaptive Processing**: Automatically identify problem types and apply specialized techniques optimized for each domain (math, coding, writing, science, etc.).

3. **Robust Answer Generation**: Implement a self-consistency approach that generates multiple solution pathways and uses intelligent voting to identify the most reliable answers.

4. **Dynamic Verification**: Apply sophisticated meta-cognitive feedback loops that analyze, critique, and refine candidate solutions, particularly when confidence is low.

5. **Parameter Optimization**: Dynamically select optimal model parameters based on problem type to maximize model performance for specific tasks.

### Technical Implementation

Project Tandem employs several novel techniques:

#### 1. Problem Classification and Parameter Optimization

The system begins by classifying the problem type (mathematical, scientific, creative writing, etc.) and dynamically configures the LLM with optimized parameters:

- **Adaptive Parameters**: Custom-tailored temperature, top-p, top-k, and other parameters specifically optimized for each problem domain
- **Specialized Model Creation**: Dynamically creates temporary models with domain-specific configurations

#### 2. Self-Consistency Solution Generation

Rather than relying on a single solution path, Tandem generates multiple diverse approaches:

- **Diverse Solution Strategies**: Employs varied prompting techniques to explore different reasoning pathways
- **Domain-Specific Prompts**: Uses specialized prompts tailored to the problem type (e.g., algebraic, computational, and visual approaches for math problems)
- **Majority Voting**: Extracts and normalizes final answers, then employs intelligent voting to identify the most consistent solution

#### 3. Advanced Answer Extraction and Normalization

The framework implements sophisticated techniques to identify and standardize answers:

- **Pattern Recognition**: Uses domain-specific regex patterns to reliably extract answers from different formats
- **Mathematical Normalization**: Implements specialized normalization for mathematical expressions, handling LaTeX, parentheses, and various formatting styles
- **Confidence Assessment**: Calculates agreement percentages and confidence levels based on solution convergence

#### 4. Adaptive Meta-Cognitive Feedback

When solution confidence is low, Tandem employs an advanced refinement process:

- **Problem-Specific Evaluation**: Applies specialized evaluation criteria based on problem domain
- **Structured Criticism**: Systematically identifies issues in factual accuracy, reasoning quality, clarity, and completeness
- **Incremental Refinement**: Applies dynamic iterations of feedback and improvement, with iteration count based on confidence levels
- **Verification**: Implements a separate verification stage with specialized models to validate final answers

### Key Innovations and Contributions

Project Tandem introduces several notable innovations:

1. **Integration of Multiple Enhancement Techniques**: Unlike approaches that rely on a single enhancement method, Tandem combines classification, parameter optimization, self-consistency, and meta-cognitive feedback into a unified pipeline.

2. **Dynamic Adaptation**: The framework automatically adapts its strategy based on problem type and confidence levels, applying different techniques only when needed.

3. **Non-Destructive Enhancement**: All improvements are achieved without modifying the base model, making the approach compatible with any local LLM.

4. **Mathematical Excellence**: Particularly strong performance on mathematical problems through specialized prompting, answer extraction, and verification techniques.

5. **Extensible Framework**: The modular design allows for easy addition of new problem domains, techniques, and optimizations.

### Results and Performance

Tandem significantly improves local LLM performance across various domains:

- **Mathematical Problem Solving**: Enhanced accuracy through specialized parameter settings, diverse solution approaches, and rigorous verification
- **Consistent Output Quality**: Reduced variability and increased reliability of responses
- **Error Detection**: Improved ability to catch and correct errors through the multi-stage verification process
- **Confidence Assessment**: Transparent reporting of solution confidence and agreement metrics

### Conclusion

Project Tandem represents a sophisticated framework for enhancing locally-deployed LLMs without requiring model modifications or fine-tuning. By intelligently orchestrating multiple techniques—problem classification, parameter optimization, diverse solution generation, and adaptive meta-cognitive feedback—Tandem enables local models to produce higher quality, more reliable responses across a wide range of problem domains.

This framework addresses a significant challenge in the AI industry by empowering users to achieve enhanced performance from their locally-run models, bridging the gap between local deployment and cloud-based capabilities.

### Technologies Used

- Python 3.7+
- Ollama API for local model interaction
- Regular expressions for pattern extraction
- Dynamic model parameter configuration
- Advanced prompt engineering techniques

### User Interface Features

Project Tandem offers an enhanced user experience with the following features:

- **Colorful Terminal Output**: Color-coded responses and status messages for better readability
- **Interactive Mode**: Conveniently enter problems directly in the terminal
- **Progress Indicators**: Clear step-by-step indicators showing the solution process
- **Solution Visualization**: Visual representation of answer agreement and distribution
- **Styled Output**: Boxed final solutions and formatted results for easier reading

### Usage

Run the tandem_runner.py script to solve problems using the Project Tandem framework:

```bash
python tandem_runner.py [options]
```

By default, the script will prompt you to enter your question or problem. You can also provide a problem in a text file and use the --input-file flag to specify the file location.

Available options:
- `--help`: Display help information about command-line options
- `--input-file PATH`: Read the problem from a specific file
- `--file`: Read the problem from prompt.txt (legacy option)
- `--iterations N`: Number of solutions to generate for self-consistency (default: 3)
- `--confidence THRESHOLD`: Confidence threshold below which to apply enhancement techniques (default: 80.0)
- `--meta`: Use meta-cognitive feedback for low confidence answers (default: enabled)
- `--no-meta`: Disable meta-cognitive feedback
- `--meta-iterations N`: Maximum number of meta-cognitive iterations (default: 3)
- `--hybridize`: Enable solution hybridization to combine multiple approaches (default: enabled)
- `--no-hybridize`: Disable solution hybridization

Examples:
```bash
# Interactive mode (prompts for input)
python tandem_runner.py

# Read from a specific file
python tandem_runner.py --input-file path/to/problem.txt

# Advanced usage with custom parameters
python tandem_runner.py --iterations 5 --confidence 70 --meta-iterations 4

# Disable solution hybridization but keep meta-cognitive feedback
python tandem_runner.py --no-hybridize

# Use only self-consistency without additional enhancements
python tandem_runner.py --no-meta --no-hybridize
```

### Output Interpretation

Tandem provides rich visual feedback on the solution process:

- **Problem Classification**: Shows detected problem category and model configuration
- **Solution Agreement**: Visual indicator of agreement percentage among different solutions
- **Answer Distribution**: Text-based bar chart showing the distribution of answers
- **Verification Status**: Color-coded verification results with feedback
- **Meta-cognitive Details**: When applicable, shows refinement iterations and improvements made

### Version History

#### v0.2.0 (February 2025)

This release introduces major enhancements to improve the framework's performance and user experience:

**Core Performance Improvements:**

1. **Progressive Temperature Control**
   - Implemented stage-specific parameter optimization for different phases of problem-solving
   - Added exploration, calculation, and verification stages with tailored parameters
   - Customized parameter profiles for different problem domains (math, coding, writing, science)
   - Enabled dynamic temperature adjustment based on problem complexity

2. **Quality-Based Weighted Voting**
   - Enhanced solution selection with sophisticated quality scoring
   - Added weighting based on verification steps, structured format, and reasoning quality
   - Implemented quality factor analysis (verification steps, reasoning depth, calculations)
   - Added detailed scoring visualization in the console output

3. **Targeted Meta-Cognitive Feedback**
   - Redesigned the feedback system with domain-specific critique templates
   - Implemented progressive focus areas (accuracy → completeness → presentation)
   - Added staged refinement process that addresses different aspects in each iteration
   - Enhanced final verification stage with comprehensive solution checking

4. **Solution Hybridization**
   - Added capability to combine the best elements from multiple solution approaches
   - Implemented intelligent segmentation of solution components
   - Added comparative evaluation to ensure hybrid solutions are superior
   - Created domain-specific hybridization strategies for different problem types

**User Experience Improvements:**

1. **Enhanced Console Readability**
   - Completely redesigned console output with improved vertical spacing
   - Added clear visual separation between process stages
   - Improved heading and section organization for better readability
   - Enhanced progress tracking with better visual indicators

2. **Detailed Process Visualization**
   - Added comprehensive solution quality metrics display
   - Enhanced meta-cognitive process reporting
   - Implemented detailed hybridization information
   - Added clearer iteration tracking and focused improvement areas

3. **Improved Command-Line Interface**
   - Added hybridization control via command-line flags
   - Enhanced configuration display at startup
   - Improved error handling and status messages
   - Added better progress indication throughout the process

#### v0.1.0 (January 2025)

Initial release of Project Tandem featuring:
- Self-consistency solution generation
- Basic meta-cognitive feedback
- Problem classification and parameter optimization
- Answer extraction and normalization