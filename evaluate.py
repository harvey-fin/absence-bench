import subprocess
from typing import *
from tqdm import tqdm
import pandas as pd
import os    
import json
import argparse
    

def run_model(model_family: str, model: str, 
              tasks:str, in_dir: str, 
              out_dir: str, batch_size: str, thinking: bool):
    """Run a single model on all tests"""
    try:
        test_files = os.listdir(in_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{in_dir}' not found.")
    if tasks:
        task_files = [f"test_llms_{t}.py" for t in tasks]
        test_files = task_files
    for file in test_files:
        if file == "llm_providers.py" or file == "__pycache__":
            continue

        # extract the specific task
        task_str = file.split("_")[-1][:-3]
        print(f"\n--- testing {model} on {task_str} tasks ---\n")

        # set the path to test files
        path_to_tests = os.path.join(in_dir, file)

        # set the path to the results directory where outputs are stored
        model_str = model
        if "/" in model_str:
            cut_idx = model_str.index("/")
            model_str = model_str[cut_idx+1:]
        if thinking:
            model_str += "_thinking"
        
        output_path = f"{task_str}_{model_str}.jsonl"
        path_to_outputs = os.path.join(out_dir, output_path)
        if os.path.exists(path_to_outputs):
            print(f"Results for {task_str} exists, skipping!\n")
            continue

        if thinking:
            with subprocess.Popen(
                ["python", path_to_tests, 
                 "--provider_models", model_family+":"+model, 
                 "--output", path_to_outputs, 
                 "--batch_size", str(batch_size), 
                 "--thinking"],
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                universal_newlines=True) as process:
                for line in process.stdout:
                    print(line, end='')
        else:
            with subprocess.Popen(
                ["python", path_to_tests, 
                 "--provider_models", model_family+":"+model, 
                 "--output", path_to_outputs, 
                 "--batch_size", str(batch_size)],
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                universal_newlines=True) as process:
                for line in process.stdout:
                    print(line, end='')


def collect_result(model:str, out_dir: str, thinking: bool=False):
    """Collect the result (average accuracy) of a single model on all tasks"""
    results = []
    provider = model.split(":")[0]
    model_str = model.split(":")[1]
    if "/" in model_str:
        cut_idx = model_str.index("/")
        model_str = model_str[cut_idx+1:]
    if thinking:
        model_str += "_thinking"

    for task_str in ["poetry", "numerical", "prs"]:
        output_path = f"{task_str}_{model_str}.jsonl"
        path_to_outputs = os.path.join(out_dir, output_path)
        try:
            with open(path_to_outputs, "r") as f:
                d = json.load(f)
        except FileNotFoundError:
            print(f"Results for {model_str} does not exist, skipping")
            return []

        model = list(d[provider].keys())[0]
        mean_acc = d[provider][model]["average_accuracy"]
        
        results.append(mean_acc)

    return {model:results}


def show_outputs(results: Dict[str, List[float]]):
    """Show the results in a table"""
    results = pd.DataFrame.from_dict(results, orient="index", 
                                     columns=["poetry", "numerical", "prs"]).round(3)
    results_copy = results.copy()
    results['average'] = results_copy.mean(numeric_only=True, axis=1).round(3)
    results.loc['mean'] = results.mean().round(3)
    results[results.select_dtypes(include=['number']).columns] *= 100
    results = results.dropna(axis="columns", how="all")

    print(results)


def main():
    parser = argparse.ArgumentParser(description='Test LLMs on all Absence Bench Tasks')
    parser.add_argument('--model_family', type=str, default='openai',
                      help='the provider of LLM '
                      '(openai/togetherai/xai/anthropic/google/custom)')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini',
                      help='model to evaluate the tasks with')
    parser.add_argument('--out_dir', type=str, default='results',
                      help='Path to the output directory to store the results')
    parser.add_argument('--in_dir', type=str, default='tests',
                      help='Path to the test scripts')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of API calls to batch together (default: 5)')
    parser.add_argument('--run_task', type=str, nargs='+', default=None,
                      help="Flag argument to run only on certain tasks")
    parser.add_argument('--thinking', action='store_true',
                      help="Active thinking mode")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    run_model(model_family=args.model_family,
                model=model,
                tasks=args.run_task,
                in_dir=args.in_dir,
                out_dir=args.out_dir,
                batch_size=args.batch_size,
                thinking=args.thinking)

    model = args.model_family + ":" + args.model
    results = collect_result(model=model, out_dir=args.out_dir, thinking=args.thinking)
    show_outputs(results)

if __name__ == "__main__":
    main()
