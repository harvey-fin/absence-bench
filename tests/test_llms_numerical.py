'''
This file contains all the code to test the LLMs on the artificial tasks.
'''
import numpy as np
import json
import argparse
import random
from pathlib import Path
import time
from typing import List, Dict, Any, Callable
import concurrent.futures
import os
from llm_providers import LLMProvider
from tqdm import tqdm

def load_artificial_tasks(jsonl_file: str, sample_size: int = None, min_idx: int=None, max_idx: int=None) -> List[Dict[str, Any]]:
    '''
    Load artificial tasks from a jsonl file.
    '''
    tasks = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    # Return a specific range of tasks if requested
    if min_idx is not None and max_idx is not None:
        return tasks[min_idx:max_idx]
    
    # Sub-sample if requested
    if sample_size and sample_size < len(tasks):
        return random.sample(tasks, sample_size)
    return tasks

def evaluate_response(response_list: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Evaluate the model's response to determine if it correctly identifies the omitted numbers.
    This should calculate micro f1 scores for the model's response.
    '''
    og_sequence = task_data['original_sequence']
    omitted_indices = np.where(~np.array(task_data['omitted_mask']))[0]

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
                
def process_artificial_task(artificial_task: Dict[str, Any], task_idx: int, total_tasks: int,
                            model_provider: str, model_name: str, system_prompt: str, thinking:bool,) -> Dict[str, Any]:
    '''
    Process a single artificial task with the given model.
    '''
    # print(f"Testing {model_provider}/{model_name} - Task {task_idx+1}/{total_tasks}")

    if artificial_task["metadata"]["task_type"] == "numerical":
        user_task_specification = 'numbers'
    elif artificial_task["metadata"]["task_type"] == "dates":
        user_task_specification = 'dates'
    
    base_sequence_stringified = '\n'.join(str(x) for x in artificial_task['original_sequence'])
    user_sequence_stringified = '\n'.join(str(x) for x in artificial_task['user_sequence'])

    user_message = f'''Here is a sequence of {user_task_specification}:

{base_sequence_stringified}

Now, here is my recitation of the sequence which may be missing some {user_task_specification}:

{user_sequence_stringified}

What {user_task_specification} did I miss? Please list only the missing {user_task_specification}, nothing else.'''

    try:
        # Get the appropriate provider
        provider = LLMProvider.get_provider(model_provider)
        
        # Get response from the provider
        response = provider.get_response(system_prompt, user_message, model_name, thinking=thinking)
        
        evaluation = evaluate_response(response, artificial_task)
        evaluation["task_id"] = artificial_task["id"]
        evaluation["model_response"] = response
        return evaluation
    
    except Exception as e:
        print(f"Error processing task {task_idx+1}: {e}")
        return None

def test_model(tasks: List[Dict[str, Any]], model_provider: str, model_name: str, batch_size: int = 5, thinking:bool=False) -> Dict[str, Any]:
    '''
    Test a model on all the tasks and return the results, processing in batches
    '''
    results = []
    total_tasks = len(tasks)

    if tasks[0]['metadata']['task_type'] == 'numerical':
        user_task_specification = 'numbers'
    elif tasks[0]['metadata']['task_type'] == 'dates':
        user_task_specification = 'dates'
    
    system_prompt = f"""You are helping a student practice reciting sequences. 
The student will recite a sequence, but they may have missed some {user_task_specification}. 
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
                    process_artificial_task, task, task_idx, total_tasks, 
                    model_provider, model_name, system_prompt, thinking)
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
    parser = argparse.ArgumentParser(description='Test LLMs on their ability on numerical tasks')
    parser.add_argument('--tasks_file', type=str, default='data/numerical_default.jsonl',
                      help='Path to the tasks JSONL file')
    parser.add_argument('--sample_size', type=int,
                      help='Number of tasks to sample for testing (default: use all)')
    parser.add_argument('--provider_models', type=str, nargs='+', default=['openai:o1-2024-12-17'],
                      help='Provider and model pairs in the format "provider:model" (e.g., "openai:gpt-4 anthropic:claude-3-opus")')
    parser.add_argument('--output', type=str, default='results/llm_numerical_test_results.json',
                      help='Path to save the test results (default: results/llm_numerical_test_results.json)')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of API calls to batch together (default: 5)')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                      help='Path to the checkpoint file to load (default: None)')
    parser.add_argument('--min_idx', type=int, default=None,
                      help='Minimum index of tasks to load (default: None)')
    parser.add_argument('--max_idx', type=int, default=None,
                      help='Maximum index of tasks to load (default: None)')
    parser.add_argument("--thinking", action='store_true',
                      help="Whether to enable the thinking mode or not")
    parser.add_argument("--check_omitted", action="store_true",
                      help="check whether all instances are omitted in previous runs"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    tasks_path = Path(args.tasks_file)
    if not tasks_path.exists():
        print(f"Error: Tasks file '{args.tasks_file}' does not exist!")
        return
    
    # Load and potentially sub-sample the tasks
    tasks = load_artificial_tasks(args.tasks_file, args.sample_size, args.min_idx, args.max_idx)
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
    if args.checkpoint_file:
        try:
            with open(args.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            checkpoint_data['test_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
        except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
            pass
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
            
        print(f"\nTesting provider: {provider}, model: {model}")
        try:
            results = test_model(tasks, provider, model, args.batch_size, args.thinking)

            # Store results by model name under the provider
            all_results[provider][model] = results
            
            print(f"{provider} ({model}): {results['average_accuracy']:.2%} average Micro F1 score")
        except Exception as e:
            print(f"Error testing {provider}/{model}: {str(e)}")
            all_results[provider][model] = {"error": str(e)}
    
        if args.checkpoint_file:
            try:
                checkpoint_data[provider][model]['detailed_results'].extend(results['detailed_results'])
                checkpoint_data[provider][model]['average_accuracy'] = sum(r['micro_f1'] for r in checkpoint_data[provider][model]['detailed_results']) / len(checkpoint_data[provider][model]['detailed_results'])
                checkpoint_data[provider][model]['total_tasks'] = len(checkpoint_data[provider][model]['detailed_results'])
            # in the case we get an empty checkpoint file and checkpoint_data is not defined.
            except UnboundLocalError:
                pass
    
    # Save results (with checkpoints if provided)
    with open(args.output, 'w', encoding='utf-8') as f:
        if args.checkpoint_file:
            try:
                json.dump(checkpoint_data, f, indent=2)
            except UnboundLocalError: # empty checkpoint file
                json.dump(all_results, f, indent=2)
        else:
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


def run_omitted(model_family: str, model:str, batch_size:int, thinking:bool):
    """function to run the datapoints that had a internet connecction error"""
    model_str = model
    if "/" in model_str:
        cut_idx = model_str.index("/")
        model_str = model_str[cut_idx+1:]
    results_file = [f"results/numerical_{model_str}.jsonl", f"results/numerical_{model_str}_thinking.jsonl"][thinking]
    alt_file =  [f"results/numerical_{model_str}_alt.jsonl", f"results/numerical_{model_str}_thinking_alt.jsonl"][thinking]
    with open(results_file, "r") as f:
        d = json.load(f)
    source_file = "data/numerical_default.jsonl"
    result_id = "task_id"
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
    orig_length = d[model_family][model]["total_tasks"] - length
    d[model_family][model]["average_accuracy"] = (orig_accs * orig_length + length * results["average_accuracy"]) / (orig_length + length)
    d[model_family][model]["average_thinking_tokens"] = (orig_thinking * orig_length + length * results["average_thinking_tokens"]) / (orig_length + length)

    with open(alt_file, 'w', encoding="utf-8") as f_out:
        json.dump(d, f_out, indent=2)
    
    print(f"saving results to {alt_file}")


    
if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     x = load_artificial_tasks('data/numerical_tasks.jsonl', sample_size=10)
#     task = x[0]
#     print(task['metadata']['task_type'])
#     print('n_numbers', len(task['original_sequence']))
#     print('n_omitted', task['metadata']['n_omitted'])
#     print('omission_prob', task['metadata']['omission_prob'])
#     print('sequence_type', task['metadata']['sequence_type'])
#     print('order', task['metadata']['order'])

#     print('-'*100)
#     print('PROCESSING....')
        
    

#     processed = process_artificial_task(task, task['id'], 10, 'openai', 'gpt-4o-mini', system_prompt)
#     print('DONE PROCESSING!')
#     print('-'*100)

#     print("MICRO F1 SCORE:", processed['micro_f1'])
#     print("MODEL RESPONSE:", [line for line in processed['model_response'].split('\n') if line.strip()])
#     print("ORIGINAL SEQUENCE:", task['original_sequence'])
#     omissions = [x for x, mask in zip(task['original_sequence'], task['omitted_mask']) if not mask]
#     print("OMISSIONS:", omissions)
#     print("USER SEQUENCE:", task['user_sequence'])