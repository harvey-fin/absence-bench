# Project Setup Guide

This README provides instructions for setting up a Python virtual environment and installing project dependencies from the requirements.txt file.

## Prerequisites

- Python 3.6 or higher installed
- pip (Python package installer)

## Setting Up a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

You'll know your virtual environment is active when you see `(venv)` at the beginning of your terminal prompt.

## Installing Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### If setuptools is missing

If you encounter `NameError: name 'setuptools' is not defined`, install setuptools:

```bash
pip install setuptools
```

### Updating pip

It's a good practice to ensure pip is up to date in your virtual environment:

```bash
pip install --upgrade pip
```

### Requirements.txt not found

Make sure you're in the project's root directory where the requirements.txt file is located.

## Poetry Usage

This project provides tools for processing poetry datasets, visualizing the processed poems, and testing LLMs on their ability to identify missing lines.


### 1. Processing Poetry Data

The `dataset_construction/process_poetry.py` script transforms a poetry dataset by creating versions of poems with randomly omitted lines.

```bash
python dataset_construction/process_poetry.py <input_file> [-o OUTPUT] [-p PROB] [-m MAX_LINES]
```

Arguments:
- `--input_file`: Path to the input poetry.jsonl file
- `-o, --output`: Path to the output file (default: poetry_processed.jsonl)
- `-p, --prob`: Probability of omitting a line (default: 0.1)
- `-m, --max_lines`: Maximum number of lines to include from each poem (optional)
- `--use_needles`: Whether to insert needles (True) or omit lines (False) (default: False)
- `--use_placeholders`: Whether to use placeholders to mark the omissions

### 2. Testing Language Models

The `tests/test_llms_poetry.py` script evaluates LLMs on their ability to identify omitted lines from poems.

```bash
python test_llms_poetry.py [--poems_file POEMS_FILE] [--sample_size SAMPLE_SIZE] [--provider_models {model_provider}:{model}] [--output OUTPUT] [--batch_size BATCH_SIZE]
```

Arguments:
- `--poems_file`: Path to the processed poems JSONL file (default: data/processed_poems.jsonl)
- `--sample_size`: Number of poems to sample for testing (default: use all)
- `--provider_models`: model family and specific model to test (example: openai:gpt-4o)
- `--output`: Path to save the test results (default: llm_poem_test_results.json)
- `--batch_size`: Number of API calls to batch together (default: 5)
- `--thinking`: Whether to use the thinking mode for inference-time compute models (default: False)
- `--check_omitted`: Check whether all instances are evaluted (some might not due to API side errors)

Example:
```bash
python test_llms_poetry.py --poems_file data/poetry_default.jsonl --sample_size 20 --provider_models openai:o1-2024-12-17 --output results.jsonl
```

Note: You'll need to set up the appropriate API keys as environment variables:
- For OpenAI: Set the OPENAI_API_KEY environment variable
- For TogetherAI: Set the TOGETHER_API_KEY environment variable
- For Google: Set the GEMINI_API_KEY environment variable
- For XAI: Set the XAI_API_KEY environment variable
- For Anthropic: Set the ANTHROPIC_API_KEY environment variable 

## Numerical Sequences
### Generate Numerical Sequences Data
```bash
python dataset_construction/generate_numeric.py
```

### Testing Language Models
The `tests/test_llms_numerical.py` script evaluates LLMs on their ability to identify omitted lines from numerical sequences.

```bash
python tests/test_llms_numerical.py [--tasks_file SEQUENCE_FILE] [--sample_size SAMPLE_SIZE] [--provider_models {model_provider}:{model}] [--output OUTPUT] [--batch_size BATCH_SIZE]
```

Arguments:
- `--sequence_file`: Path to the processed numerical sequences JSONL file (default: data/numerical_default.jsonl)
- `--sample_size`: Number of numerical sequences to sample for testing (default: use all)
- `--provider_models`: model family and specific model to test (example: openai:gpt-4o)
- `--output`: Path to save the test results (default: llm_poem_test_results.json)
- `--batch_size`: Number of API calls to batch together (default: 5)
- `--thinking`: Whether to use the thinking mode for inference-time compute models (default: False)
- `--check_omitted`: Check whether all instances are evaluted (some might not due to API side errors)

Example:
```bash
python tests/test_llms_numerical.py --poems_file data/numerical_default.jsonl --sample_size 20 --provider_models openai:o1-2024-12-17 --output results.jsonl
```

## Github Merged PRs 

1. Download diffs of closed pull requests from a popular github repo 
```
$ python dataset_construction/github_merged_prs.py nodejs/node                     
Processing repository nodejs/node
Only PRs merged after 2024-04-12T22:32:02.174295+00:00 and with line changes in [10, 100] will be kept
Looking for up to 10 merges.
Processing PR #57667 (merged at 2025-04-12T18:27:10Z)
PR #57667 - +0 / -9 (total 9 changes)
    Does not meet line change criteria. Skipping.
[...]
Processing PR #57790 (merged at 2025-04-10T22:37:31Z)
PR #57790 - +16 / -30 (total 46 changes)

Collected 10 merged PRs. Saving output.
Output saved to data/merges/nodejs__node-2025-04-12-merged-prs.jsonl
```

2. Randomly delete a few lines from each diff.
*(There's also a condition to delete non-changed lines which should be ignored as 
per the models prompt---they don't change the resulting commit---but
`tests/test_llms_github_prs.py` doesn't take off points yet for this.)*
```
$ python process_github_prs.py data/merges/nodejs__node-2025-04-12-merged-prs.jsonl
Processing complete. Output saved to github_prs_processed.jsonl
```

3. Test how well LLMs can manage to notice the changes.
```
$ python tests/test_llms_github_prs.py --diffs_file github_prs_default.jsonl
Loaded 10 diff records for testing.

Testing provider: openai, model: gpt-4o
Processing batch 1 (1-5 of 10)
Testing openai/gpt-4o - Diff 1/10
Testing openai/gpt-4o - Diff 2/10
Testing openai/gpt-4o - Diff 3/10
Testing openai/gpt-4o - Diff 4/10
Testing openai/gpt-4o - Diff 5/10
Waiting a moment before the next batch...
Processing batch 2 (6-10 of 10)
Testing openai/gpt-4o - Diff 6/10
Testing openai/gpt-4o - Diff 7/10
Testing openai/gpt-4o - Diff 8/10
Testing openai/gpt-4o - Diff 9/10
Testing openai/gpt-4o - Diff 10/10
openai (gpt-4o): 40.33% average accuracy

Results saved to llm_diff_test_results.json

Summary:
openai (gpt-4o): 40.33% average accuracy
```
