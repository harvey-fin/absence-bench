"""
This script tests LLMs on their ability to recall the changes made in a sequence of numbers.
It accepts as input a processed JSON Lines file (e.g. one produced by generate_numeric.py)
containing, for each PR, the following keys:
  - "original_context": The full sequence of numbers
  - "modified_context": The sequence after randomly deleting some numbers.
  - "omitted_index": A list of the exact text of each changed number
    (insertions/deletions) that was removed.

The LLM’s response is then evaluated by checking if it correctly identifies
    the missing changed numbers.
Each evaluation includes the total expected changed lines omitted and the number and list
of those that the model correctly recalled.

Example usage:
    python test_llm_numerical.py --input_file data/numerical.jsonl
         [--sample_size N] [--provider_models openai:gpt-4 ...]
         [--output results/numerical_gpt-4.json] [--batch_size 5]
"""

import numpy as np
import json
import argparse
import random
import time
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Callable
from llm_providers import LLMProvider


def load_artificial_tasks(jsonl_file: str, sample_size: int = None) -> List[Dict[str, Any]]:
    '''
    Load artificial tasks from a jsonl file.
    '''
    tasks = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    # Sub-sample if requested
    if sample_size and sample_size < len(tasks):
        return random.sample(tasks, sample_size)
    return tasks


def evaluate_response(response_list: str, 
                      task_data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Evaluate the model's response to determine if it correctly identifies the omitted numbers.
    This should calculate micro f1 scores for the model's response.
    '''
    og_sequence = task_data['original_context'].split('\n')
    omitted_indices = task_data['omitted_index']

    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_elements": [],
        "unidentified_elements": [],
        "wrongly_identified_elements": [],
    }

    if response_list[0] == None:
        response = ""
        thinking_tokens = 0
    else:
        response = response_list[0]
        thinking_tokens = response_list[1]

    for idx, element in enumerate(og_sequence):
        str_element = str(element)
        if str_element in [x.strip() for x in response.split('\n')]:
            if idx in omitted_indices:
                results["tp"] += 1
                results["identified_elements"].append(element)
            else:
                results["fp"] += 1
                results["wrongly_identified_elements"].append(element)
        else:
            if idx in omitted_indices:
                results["fn"] += 1
                results["unidentified_elements"].append(element)
    
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0
    
    # when there are no omissions:
    if len(omitted_indices) == 0:
        results["micro_f1"] = 1-  results["fp"]/len(og_sequence)

    results["thinking_token"] = thinking_tokens
    return results

                
def process_artificial_task(
        artificial_task: Dict[str, Any], 
        task_idx: int,
        model_provider: str, 
        model_name: str, 
        system_prompt: str, 
        thinking: bool,
    ) -> Dict[str, Any]:
    '''
    Process a single artificial task with the given model.
    '''

    user_message = f'''Here is a sequence of numbers:

{artificial_task['original_context']}

Now, here is my recitation of the sequence which may be missing some numbers:

{artificial_task['modified_context']}

What numbers did I miss? Please list only the missing numbers, nothing else.'''

    try:
        # Get the appropriate provider
        provider = LLMProvider.get_provider(model_provider)
        
        # Get response from the provider
        response = provider.get_response(system_prompt, user_message, model_name, thinking=thinking)
        
        evaluation = evaluate_response(response, artificial_task)
        evaluation["id"] = artificial_task["id"]
        evaluation["model_response"] = response
        return evaluation
    
    except Exception as e:
        print(f"Error processing task {task_idx+1}: {e}")
        return None

def test_model(
        tasks: List[Dict[str, Any]], 
        model_provider: str, 
        model_name: str, 
        batch_size: int = 5, 
        thinking: bool = False
    ) -> Dict[str, Any]:
    '''
    Test a model on all the tasks and return the results, processing in batches
    '''
    results = []
    total_tasks = len(tasks)
    
    system_prompt = f"""You are helping a student practice reciting sequences. 
The student will recite a sequence, but they may have missed some numbers. 
Your task is to identify exactly which numbers are missing from their recitation.
List only the missing numbers, nothing else."""
    
    for batch_start in tqdm(range(0, total_tasks, batch_size)):
        batch_end = min(batch_start + batch_size, total_tasks)
        current_batch = tasks[batch_start:batch_end]

        # print(f"Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {total_tasks})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, task in enumerate(current_batch):
                task_idx = batch_start + i
                future = executor.submit(
                    process_artificial_task, 
                    task, 
                    task_idx,
                    model_provider, 
                    model_name, 
                    system_prompt, 
                    thinking)
                futures.append(future)
            
            # collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                
        # add a small delay between batches to avoid rate limits
        if batch_end < total_tasks:
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
        "total_tasks": total_tasks,
        "average_accuracy": avg_accuracy,
        "average_thinking_tokens": avg_thinking,
        "detailed_results": results
    }
    
def main():
    parser = argparse.ArgumentParser(
        description='Test LLMs on their ability on numerical tasks'
    )
    parser.add_argument('--input_file', type=str, default='data/numerical.jsonl',
                      help='Path to the tasks JSONL file')
    parser.add_argument('--sample_size', type=int,
                      help='Number of tasks to sample for testing (default: use all)')
    parser.add_argument('--provider_models', type=str, nargs='+', default=['openai:o1-2024-12-17'],
                      help='Provider and model pairs in the format \"provider:model\" '
                      '(e.g., "openai:gpt-4 anthropic:claude-3-opus")')
    parser.add_argument('--output', type=str,
                      help='Path to save the test results')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of API calls to batch together (default: 5)')
    parser.add_argument("--thinking", action='store_true',
                      help="Whether to enable the thinking mode or not")
    
    args = parser.parse_args()
    
    # Check if input file exists
    tasks_path = Path(args.input_file)
    if not tasks_path.exists():
        print(f"Error: Tasks file '{args.input_file}' does not exist!")
        return
    
    # Load and potentially sub-sample the tasks
    tasks = load_artificial_tasks(args.input_file, args.sample_size)
    print(f"Loaded {len(tasks)} tasks for testing")

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
    all_results = {"test_date": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Test each provider-model pair
    for provider, model in provider_models:
        # Initialize provider dictionary if it doesn't exist
        if provider not in all_results:
            all_results[provider] = {}
            
        print(f"\nTesting provider: {provider}, model: {model}")
        try:
            results = test_model(tasks, provider, model, args.batch_size, args.thinking)

            # Store results by model name under the provider
            all_results[provider][model] = results
            
            print(f"{provider} ({model}): {results['average_accuracy']:.2%} average Micro F1 score")
        except Exception as e:
            print(f"Error testing {provider}/{model}: {str(e)}")
            all_results[provider][model] = {"error": str(e)}
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("\nSummary:")
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
