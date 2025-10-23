"""
This script tests LLMs on their ability to recall the changes made in a Poem.
It accepts as input a processed JSON Lines file (e.g. one produced by process_poetry.py)
containing, for each PR, the following keys:
  - "original_context": The full poem retrieved from the Gutenberg Poetry Corpus
  - "modified_context": The poem after randomly deleting some lines.
  - "omitted_context": A list of the exact text of each changed line
    that was removed.

The LLM’s response is then evaluated by checking if it correctly identifies
    the missing changed lines.
Each evaluation includes the total expected changed lines omitted and the number and list
of those that the model correctly recalled.

Example usage:
    python test_llm_poetry.py --input_file data/poetry.jsonl
         [--sample_size N] [--provider_models openai:gpt-4 ...]
         [--output results/poetry_gpt-4.json] [--batch_size 5]
"""

import json
import argparse
import random
import time
import os
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any
from llm_providers import LLMProvider


def load_poems(
        jsonl_file: str, 
        sample_size: int = None,
    ) -> List[Dict[str, Any]]:
    """
    Load poems from the JSONL file, with optional sub-sampling
    """
    poems = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            poems.append(json.loads(line))
    
    # Sub-sample if requested
    if sample_size and sample_size < len(poems):
        return random.sample(poems, sample_size)

    return poems


def evaluate_response(
        response_list: str, 
        poem_data: Dict[str, Any],
        use_needle: bool,
    ) -> Dict[str, Any]:
    """
    Evaluate the model's response to determine if it correctly identified the omitted lines
    This is updated to micro f1 scores
    """
    original_lines = poem_data["original_context"].split('\n')
    omitted_indices = poem_data["omitted_index"]
    # if we are testing needle in a haystack, then we will not use omitted_indices for omissions
    if use_needle:
        needles = poem_data['needles']

    if response_list[0] == None:
        response = ""
        thinking_tokens = 0
    else:
        response = response_list[0]
        thinking_tokens = response_list[1]

    # some models might include thinking tokens in their response
    if "</think>" in response:
        response = response[response.index("</think>")+8:]
    
    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": []
    }

    if use_needle:
        for needle in needles:
            if needle.lower() in response.lower():
                results["tp"] += 1
                results["identified_lines"].append(needle)
            else:
                results["fn"] += 1
                results["unidentified_lines"].append(needle)
        for idx, line in enumerate(original_lines):
            clean_line = line.strip().lower()
            if clean_line and clean_line in response.lower():
                if idx not in omitted_indices:
                    results["fp"] += 1
                    results["wrongly_identified_lines"].append(line)
    else:
        for idx, line in enumerate(original_lines):
            # Clean up the line for comparison (remove punctuation, extra spaces, etc.)
            clean_line = line.strip().lower()
            if clean_line and clean_line in response.lower():
                if idx in omitted_indices:
                    results["tp"] += 1
                    results["identified_lines"].append(line)
                else:
                    results["fp"] += 1
                    results["wrongly_identified_lines"].append(line)
            elif clean_line and clean_line not in response.lower():
                results["fn"] += 1
                results["unidentified_lines"].append(line)

    # calculate micro_f1 score
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0

    if len(omitted_indices) == 0:
        results["micro_f1"] = 1-  results["fp"]/len(original_lines)

    results["thinking_token"] = thinking_tokens
    return results


def process_poem(
        poem: Dict[str, Any], 
        poem_idx: int, 
        model_provider: str, 
        model_name: str, 
        system_prompt: str, 
        thinking: bool,
        use_needle: bool,
    ) -> Dict[str, Any]:
    """Process a single poem with the given model"""
    # print(f"Testing {model_provider}/{model_name} - Poem {poem_idx+1}/{total_poems}")
    
    user_message = f"""Here is the complete original poem:

{poem['original_context']}

Now, here is my recitation which may be missing some lines:

{poem['modified_context']}

What lines did I miss? Please list only the missing lines, nothing else."""
    if use_needle:
            user_message = f"""Here is the complete original poem:
{poem['original_context']}

Now, here is my recitation with some extra lines that is related to Harry Potter novel series:

{poem['modified_context']}

What lines did I add to the poem? Please list only the extra lines, nothing else."""
    
    try:
        # Get the appropriate provider
        provider = LLMProvider.get_provider(model_provider)
        
        # Get response from the provider
        response = provider.get_response(system_prompt, user_message, model_name, thinking)
            
        evaluation = evaluate_response(response, poem, use_needle)
        evaluation["id"] = poem.get("id", poem_idx)
        evaluation["model_response"] = response
        return evaluation
        
    except Exception as e:
        print(f"Error with {model_provider}/{model_name} on poem {poem_idx}: {str(e)}")
        return None

def test_model(
        poems: List[Dict[str, Any]], 
        model_provider: str, 
        model_name: str, 
        batch_size: int = 5, 
        thinking: bool = False, 
        use_needle: bool = False,
    ) -> Dict[str, Any]:
    """
    Test a model on all the poems and return the results, processing in batches
    """
    results = []
    total_poems = len(poems)
    
    system_prompt = """You are helping a student practice memorizing poems. 
The student will recite a poem, but they may have missed some lines. 
Your task is to identify exactly which lines are missing from their recitation.
List only the missing lines, nothing else."""

    if use_needle:
        system_prompt = """You are helping a student practice memorizing poems.
The student will recite a poem, but they may have added some random lines that related to Harry Potter characters.
Your task is to identify exactly which lines are not in the original poem.
List only the extra lines, nothing else."""
    
    # Process poems in batches
    for batch_start in tqdm(range(0, total_poems, batch_size)):
        batch_end = min(batch_start + batch_size, total_poems)
        current_batch = poems[batch_start:batch_end]
        
        # Process the batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, poem in enumerate(current_batch):
                poem_idx = batch_start + i
                future = executor.submit(
                    process_poem, 
                    poem, 
                    poem_idx,
                    model_provider, 
                    model_name, 
                    system_prompt, 
                    thinking, 
                    use_needle
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Add a small delay between batches to avoid rate limits
        if batch_end < total_poems:
            # print(f"Waiting a moment before the next batch...")
            time.sleep(2)
    
    # Calculate aggregate statistics
    accuracy_sum = sum(r["micro_f1"] for r in results)
    avg_accuracy = accuracy_sum / len(results) if results else 0
    thinking_sum = sum(r["thinking_token"] for r in results)
    avg_thinking = thinking_sum / len(results) if results else 0
    
    return {
        "model_provider": model_provider,
        "model_name": model_name,
        "total_poems": len(poems),
        "average_accuracy": avg_accuracy,
         "average_thinking_tokens": avg_thinking,
        "detailed_results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test LLMs on their ability to identify omitted lines from poems'
    )
    parser.add_argument('--input_file', type=str, default='data/poetry.jsonl',
                      help='Path to the processed poems JSONL file')
    parser.add_argument('--sample_size', type=int,
                      help='Number of poems to sample for testing (default: use all)')
    parser.add_argument('--provider_models', type=str, nargs='+', default=['openai:o1-2024-12-17'],
                      help='Provider and model pairs in the format "provider:model" '
                      '(e.g., "openai:gpt-4 anthropic:claude-3-opus")')
    parser.add_argument('--output', type=str,
                      help='Path to save the test results')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of API calls to batch together (default: 5)')
    parser.add_argument("--use_needle", action="store_true",
                      help='evalute with the NIAH setting')
    parser.add_argument("--thinking", action='store_true',
                      help="Whether to enable the thinking mode or not")
    
    args = parser.parse_args()
    
    # Check if input file exists
    poems_path = Path(args.input_file)
    if not poems_path.exists():
        print(f"Error: Poems file '{args.input_file}' does not exist!")
        return
    
    # Load and potentially sub-sample the poems
    poems = load_poems(args.input_file, args.sample_size)
    print(f"Loaded {len(poems)} poems for testing")
    
    # Parse provider:model pairs
    provider_models = []
    for pair in args.provider_models:
        if ':' not in pair:
            print(f"Warning: Skipping invalid provider-model pair '{pair}'. Format should be 'provider:model'")
            continue
        provider, model = pair.split(':', 1)
        provider_models.append((provider, model))
    
    if not provider_models:
        print("Error: No valid provider-model pairs specified!")
        return
    
    # Initialize results dictionary
    all_results = {
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Test each provider-model pair
    for provider, model in provider_models:
        # Initialize provider dictionary if it doesn't exist
        if provider not in all_results:
            all_results[provider] = {}
            
        print(f"Testing provider: {provider}, model: {model}")
        try:
            results = test_model(poems, provider, model, 
                                 args.batch_size, args.thinking, args.use_needle)
            
            # Store results by model name under the provider
            all_results[provider][model] = results
            
            print(f"{provider} ({model}): {results['average_accuracy']:.2%} average Micro F1 score")
        except Exception as e:
            print(f"Error testing {provider}/{model}: {str(e)}")
            all_results[provider][model] = {"error": str(e)}
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print("Summary:")
    for provider in all_results:
        if provider == "test_date":
            continue
        for model, results in all_results[provider].items():
            if "average_accuracy" in results:
                print(f"{provider} ({model}): {results['average_accuracy']:.2%} average Micro F1 score")
            else:
                print(f"{provider} ({model}): Error - {results.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()