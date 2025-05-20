"""
This script tests LLMs on their ability to recall the changes made in a Poem.
It accepts as input a processed JSON Lines file (e.g. one produced by process_poetry.py)
containing, for each PR, the following keys:
  - "original_poem": The full diff as retrieved from GitHub.
  - "modified_poem": The diff after randomly deleting some lines.
  - "omitted_line_indices": A list of indices of the exact text of each changed line
    (insertions/deletions) that was removed.

The LLM is prompted with both the original and the modified diff. The system prompt appears below

The LLM’s response is then evaluated by checking if it correctly identifies
    the missing changed lines.
Each evaluation includes the total expected changed lines omitted and the number and list
of those that the model correctly recalled.

Usage:
    python test_llm_poetry.py --diffs_file path_to_processed_poetry
         [--sample_size N] [--provider_models openai:gpt-4 ...]
         [--output llm_diff_test_results.json] [--batch_size 5]
         [optional: --thinking True] [optional --use_needle]
"""

import json
import argparse
import random
from pathlib import Path
import time
from typing import List, Dict, Any, Callable
import concurrent.futures
from llm_providers import LLMProvider
from tqdm import tqdm

def load_poems(jsonl_file: str, sample_size: int = None) -> List[Dict[str, Any]]:
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

def evaluate_response(response_list: str, poem_data: Dict[str, Any], use_needle:bool) -> Dict[str, Any]:
    """
    Evaluate the model's response to determine if it correctly identified the omitted lines
    This is updated to micro f1 scores
    """
    original_lines = poem_data["original_poem"].split('\n')
    omitted_indices = poem_data["omitted_line_indices"]
    # if we are testing needle in a haystack, then we will not use omitted_indices for omissions
    if use_needle:
        omitted_indices = []
        needles = poem_data['needles']

    if response_list[0] == None:
        response = ""
        thinking_tokens = 0
    else:
        response = response_list[0]
        thinking_tokens = response_list[1]
    
    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": []
    }
    
    #TODO:
    #There might be some bugs here in calculating f1 scores
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
            if idx in omitted_indices:
                results["fn"] += 1
                results["unidentified_lines"].append(line)
    if use_needle:
        for needle in needles:
            if needle.lower() in response.lower():
                results["tp"] += 1
                results["identified_lines"].append(needle)
            else:
                results["fn"] += 1
                results["unidentified_lines"].append(needle)

    # calculate micro_f1 score
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0

    if len(omitted_indices) == 0:
        results["micro_f1"] = 1-  results["fp"]/len(original_lines)

    results["thinking_token"] = thinking_tokens
    return results

def process_poem(poem: Dict[str, Any], poem_idx: int, total_poems: int, 
                model_provider: str, model_name: str, system_prompt: str, thinking: bool, use_needle:bool,) -> Dict[str, Any]:
    """Process a single poem with the given model"""
    # print(f"Testing {model_provider}/{model_name} - Poem {poem_idx+1}/{total_poems}")
    
    user_message = f"""Here is the complete original poem:

{poem['original_poem']}

Now, here is my recitation which may be missing some lines:

{poem['modified_poem']}

What lines did I miss? Please list only the missing lines, nothing else."""
    if use_needle:
        user_message = f"""Here is the complete original poem:
{poem['original_poem']}

Now, here is my recitation with some extra lines that is related to Harry Potter novel series:

{poem['modified_poem']}

What lines did I add to the poem? Please list only the extra liens, nothing else."""
    
    try:
        # Get the appropriate provider
        provider = LLMProvider.get_provider(model_provider)
        
        # Get response from the provider
        response = provider.get_response(system_prompt, user_message, model_name, thinking)
            
        evaluation = evaluate_response(response, poem, use_needle)
        evaluation["poem_id"] = poem.get("id", poem_idx)
        evaluation["model_response"] = response
        return evaluation
        
    except Exception as e:
        print(f"Error with {model_provider}/{model_name} on poem {poem_idx}: {str(e)}")
        return None

def test_model(poems: List[Dict[str, Any]], model_provider: str, model_name: str, batch_size: int = 5, thinking:bool=False, use_needle:bool=False) -> Dict[str, Any]:
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
        
        # print(f"Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {total_poems})")
        
        # Process the batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, poem in enumerate(current_batch):
                poem_idx = batch_start + i
                future = executor.submit(
                    process_poem, poem, poem_idx, total_poems, 
                    model_provider, model_name, system_prompt, thinking, use_needle
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
    parser = argparse.ArgumentParser(description='Test LLMs on their ability to identify omitted lines from poems')
    parser.add_argument('--poems_file', type=str, default='data/poetry_default.jsonl',
                      help='Path to the processed poems JSONL file')
    parser.add_argument('--sample_size', type=int,
                      help='Number of poems to sample for testing (default: use all)')
    parser.add_argument('--provider_models', type=str, nargs='+', default=['openai:o1-2024-12-17'],
                      help='Provider and model pairs in the format "provider:model" (e.g., "openai:gpt-4 anthropic:claude-3-opus")')
    parser.add_argument('--output', type=str, default='llm_poem_test_results.json',
                      help='Path to save the test results (default: llm_poem_test_results.json)')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of API calls to batch together (default: 5)')
    parser.add_argument("--thinking", action='store_true',
                      help="Whether to enable the thinking mode or not")
    parser.add_argument("--check_omitted", action="store_true",
                      help="check whether all instances are omitted in previous runs")
    parser.add_argument("--use_needle", action="store_true",
                      help='evalute with the NIAH setting')
    
    args = parser.parse_args()
    
    # Check if input file exists
    poems_path = Path(args.poems_file)
    if not poems_path.exists():
        print(f"Error: Poems file '{args.poems_file}' does not exist!")
        return
    
    # Load and potentially sub-sample the poems
    poems = load_poems(args.poems_file, args.sample_size)
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
        
        if args.check_omitted:
            run_omitted(provider, model, args.batch_size, args.thinking)
            return
            
        print(f"Testing provider: {provider}, model: {model}")
        try:
            results = test_model(poems, provider, model, args.batch_size, args.thinking, args.use_needle)
            
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


def run_omitted(model_family: str, model:str, batch_size:int, thinking:bool):
    """function to run the datapoints that had a internet connecction error"""
    model_str = model
    if "/" in model_str:
        cut_idx = model_str.index("/")
        model_str = model_str[cut_idx+1:]
    results_file = [f"results/poetry_{model_str}.jsonl", f"results/poetry_{model_str}_thinking.jsonl"][thinking]
    alt_file =  [f"results/poetry_{model_str}_alt.jsonl", f"results/poetry_{model_str}_thinking_alt.jsonl"][thinking]
    with open(results_file, "r") as f:
        d = json.load(f)
    source_file = "data/poetry_default.jsonl"
    result_id = "poem_id"
    id_str = "id"
    model = list(d[model_family].keys())[0]
    details = d[model_family][model]["detailed_results"]
    ids = [prs[result_id] for prs in details]
    source_idx = []
    tasks = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            source_idx.append(line[id_str])
            tasks.append(line)
    omit_idx = [i for i in source_idx if i not in ids]
    if len(omit_idx) == 0:
        print("all datapoints are evaluated!")
        return
    
    omitted_tasks = [task for task in tasks if task[id_str] in omit_idx]
    length = len(omitted_tasks)
    print(f"Evaluting a total of {length} tasks")
    results = test_model(omitted_tasks, model_family, model, batch_size, thinking)
    d[model_family][model]["detailed_results"] += results["detailed_results"]
    orig_accs = d[model_family][model]["average_accuracy"]
    orig_thinking = d[model_family][model]["average_thinking_tokens"]
    orig_length = d[model_family][model]["total_poems"] - length
    d[model_family][model]["average_accuracy"] = (orig_accs * orig_length + length * results["average_accuracy"]) / (orig_length + length)
    d[model_family][model]["average_thinking_tokens"] = (orig_thinking * orig_length + length * results["average_thinking_tokens"]) / (orig_length + length)

    with open(alt_file, 'w', encoding="utf-8") as f_out:
        json.dump(d, f_out, indent=2)
    
    print(f"saving results to {alt_file}")


if __name__ == "__main__":
    main()