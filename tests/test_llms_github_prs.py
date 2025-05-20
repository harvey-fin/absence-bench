"""
This script tests LLMs on their ability to recall the changes made in a GitHub merge PR diff.
It accepts as input a processed JSON Lines file (e.g. one produced by process_github_prs.py)
containing, for each PR, the following keys:
  - "original_diff": The full diff as retrieved from GitHub.
  - "modified_diff": The diff after randomly deleting some lines.
  - "omitted_lines": A list of the exact text of each changed line
    (insertions/deletions) that was removed.

The LLM is prompted with both the original and the modified diff. The system prompt appears below

The LLM’s response is then evaluated by checking if it correctly identifies
    the missing changed lines.
Each evaluation includes the total expected changed lines omitted and the number and list
of those that the model correctly recalled.

Usage:
    python test_llm_github_prs.py --diffs_file data/github_prs_processed.jsonl
         [--sample_size N] [--provider_models openai:gpt-4 ...]
         [--output llm_diff_test_results.json] [--batch_size 5]
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
import random
import time
from tqdm import tqdm
from typing import List, Dict, Any, Union

from llm_providers import (
    LLMProvider,
)  # Ensure this module is available in your PYTHONPATH


def load_diffs(jsonl_file: str, sample_size: int = None) -> List[Dict[str, Any]]:
    """
    Load diff records from the JSONL file. Optionally sub-sample to a given number.
    Each record is expected to have keys like
        "original_diff", "modified_diff", and "omitted_lines".
    """
    diffs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            diffs.append(json.loads(line))

    if sample_size and sample_size < len(diffs):
        return random.sample(diffs, sample_size)
    return diffs


def evaluate_response(response_list: List[Union[str, int]], diff_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the model's response to determine if it correctly identified
        the omitted changed lines.
    The expected omitted changed lines come from the "omitted_lines" list in diff_data.
    The evaluation checks if the response (after cleaning) contains the non-omitted lines and the omitted lines
    """

    original_lines = diff_data["original_diff"].split('\n')
    omitted_indices = diff_data["omitted_index"]
    
    results = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "identified_lines": [],
        "unidentified_lines": [],
        "wrongly_identified_lines": []
    }
    if response_list[0] == None:
        response = ""
        thinking_tokens = 0
    else:
        response = response_list[0]
        thinking_tokens = response_list[1]
    
    #TODO:
    repeat_lines =  list(set([l for l in original_lines if original_lines.count(l) != 1]))
    for line in repeat_lines:
        line_count =  min(response.lower().count("\n"+line.strip().lower()+"\n"), original_lines.count(line))
        results["fp"] += line_count
        for i in range(line_count):
            results["wrongly_identified_lines"].append(line)
    #There might be some bugs here in calculating f1 scores
    for idx, line in enumerate(original_lines):
        if line in repeat_lines:
            continue
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
            
    # calculate micro_f1 score
    try:
        results["micro_f1"] = 2*results["tp"] / (2*results["tp"]+results["fp"]+results["fn"])
    except Exception as e:
        results["micro_f1"] = 0
     
    if len(omitted_indices) == 0:
        results["micro_f1"] = 1-  results["fp"]/len(original_lines)
    
    results["thinking_token"] = thinking_tokens

    return results


def process_diff(
    diff_record: Dict[str, Any],
    diff_idx: int,
    total_diffs: int,
    model_provider: str,
    model_name: str,
    system_prompt: str,
    thinking: bool,
) -> Dict[str, Any]:
    """
    Process a single diff record with the specified model.
    The prompt is constructed using the original and modified diffs from the record.
    The LLM is asked to list the missing
    changed lines (insertions/deletions) from the original diff.
    """
    # print(f"Testing {model_provider}/{model_name} - Diff {diff_idx+1}/{total_diffs}")

    # Construct the user prompt using the provided data.
    user_message = f"""Here is the complete original diff:

{diff_record['original_diff']}

And here is the merge diff after the developer fixed the commit history:

{diff_record['modified_diff']}

What changed lines (insertions or deletions) present \
in the original diff are missing in the merge diff (if any)?
List only the missing changed lines, nothing else."""

    try:
        # Retrieve the LLM provider based on model_provider.
        provider = LLMProvider.get_provider(model_provider)

        # Get the model response.
        response = provider.get_response(system_prompt, user_message, model_name, thinking=thinking)

        evaluation = evaluate_response(response, diff_record)
        evaluation["diff_id"] = diff_record.get("pr_number", diff_idx)
        evaluation["model_response"] = response
        return evaluation

    except Exception as e:
        print(f"Error with {model_provider}/{model_name} on diff {diff_idx}: {str(e)}")
        return None


def test_model(
    diffs: List[Dict[str, Any]],
    model_provider: str,
    model_name: str,
    batch_size: int = 5,
    thinking: bool = False,
) -> Dict[str, Any]:
    """
    Test a model on the provided diffs (processed JSONL records) in batches.
    Returns aggregate statistics and detailed results per diff.
    """
    results = []
    total_diffs = len(diffs)

    system_prompt = (
        "You are helping a software developer determine if their merge"
        " of a pull request was successful. "
        "The developer had to edit the commit history and just wants to make sure"
        " that they have not changed what will be merged. "
        "They will list the changed lines. "
        "Your job is to figure out if they have missed any "
        "insertions or deletions from the original merge. "
        "Only pay attention to the insertions and deletions (ignore the context of the diff)."
    )

    for batch_start in tqdm(range(0, total_diffs, batch_size)):
        batch_end = min(batch_start + batch_size, total_diffs)
        current_batch = diffs[batch_start:batch_end]

        # print(
        #     f"Processing batch {batch_start//batch_size + 1} "
        #     f"({batch_start+1}-{batch_end} of {total_diffs})"
        # )

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, diff_record in enumerate(current_batch):
                diff_idx = batch_start + i
                future = executor.submit(
                    process_diff,
                    diff_record,
                    diff_idx,
                    total_diffs,
                    model_provider,
                    model_name,
                    system_prompt,
                    thinking,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        if batch_end < total_diffs:
            # print("Waiting a moment before the next batch...")
            time.sleep(2)

    accuracy_sum = sum(r["micro_f1"] for r in results)
    avg_accuracy = accuracy_sum / len(results) if results else 0
    thinking_sum = sum(r["thinking_token"] for r in results)
    avg_thinking = thinking_sum / len(results) if results else 0

    return {
        "model_provider": model_provider,
        "model_name": model_name,
        "total_diffs": total_diffs,
        "average_accuracy": avg_accuracy,
        "average_thinking_tokens": avg_thinking,
        "detailed_results": results,
    }


def main():
    """
    The main function
    """
    parser = argparse.ArgumentParser(
        description=(
            "Test LLMs on their ability to recall omitted changed"
            " lines from GitHub merge PR diffs"
        )
    )
    parser.add_argument(
        "--diffs_file",
        type=str,
        default="data/github_prs_default.jsonl",
        help="Path to the processed GitHub PR diffs JSONL file",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Number of diff records to sample for testing (default: use all)",
    )
    parser.add_argument(
        "--provider_models",
        type=str,
        nargs="+",
        default=["openai:gpt-4o"],
        help=(
            'Provider and model pairs in the format "provider:model"'
            ' (e.g., "openai:gpt-4o anthropic:claude-3-opus")'
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/llm_diff_test_results.json",
        help="Path to save the test results (default: llm_diff_test_results.json)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of API calls to batch together (default: 5)",
    )
    parser.add_argument(
        "--thinking",
        action='store_true',
        help="Whether to enable the thinking mode or not"
    )
    parser.add_argument(
        "--check_omitted",
        action="store_true",
        help="check whether all instances are omitted in previous runs"
    )

    args = parser.parse_args()

    diffs_path = Path(args.diffs_file)
    if not diffs_path.exists():
        print(f"Error: Diffs file '{args.diffs_file}' does not exist!")
        return

    diffs = load_diffs(args.diffs_file, args.sample_size)
    print(f"Loaded {len(diffs)} diff records for testing.")

    # Parse provider:model pairs.
    provider_models = []
    for pair in args.provider_models:
        if ":" not in pair:
            print(
                f"Warning: Skipping invalid provider-model pair '{pair}'."
                " Format should be 'provider:model'"
            )
            continue
        provider, model = pair.split(":", 1)
        provider_models.append((provider, model))

    if not provider_models:
        print("Error: No valid provider-model pairs specified!")
        return

    all_results = {"test_date": time.strftime("%Y-%m-%d %H:%M:%S")}

    for provider, model in provider_models:
        if provider not in all_results:
            all_results[provider] = {}

        if args.check_omitted:
            run_omitted(provider, model, args.thinking)
            return

        print(f"Testing provider: {provider}, model: {model}")
        try:
            results = test_model(diffs, provider, model, args.batch_size, args.thinking)
            all_results[provider][model] = results
            print(
                f"{provider} ({model}): {results['average_accuracy']:.2%} Micro F1"
            )
        except Exception as e:
            print(f"Error testing {provider}/{model}: {str(e)}")
            all_results[provider][model] = {"error": str(e)}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {args.output}\n")
    print("Summary:")
    for provider in all_results:
        if provider == "test_date":
            continue
        for model, results in all_results[provider].items():
            if "average_accuracy" in results:
                print(
                    f"{provider} ({model}): {results['average_accuracy']:.2%} Micro F1"
                )
            else:
                print(
                    f"{provider} ({model}): Error - {results.get('error', 'unknown error')}"
                )


def run_omitted(model_family: str, model:str, thinking:bool):
    """function to run the datapoints that had a internet connecction error"""
    model_str = model
    if "/" in model_str:
        cut_idx = model_str.index("/")
        model_str = model_str[cut_idx+1:]
    results_file = [f"results/prs_{model_str}.jsonl", f"results/prs_{model_str}_thinking.jsonl"][thinking]
    with open(results_file, "r") as f:
        d = json.load(f)
    source_file = "data/github_prs_default.jsonl"
    result_id = "diff_id"
    id_str = "issue_id"
    model = list(d[model_family].keys())[0]
    details = d[model_family][model]["detailed_results"]
    ids = [prs[result_id] for prs in details]
    source_idx = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            source_idx.append(line[id_str])
    omit_idx = [i for i in source_idx if i not in ids]
    if len(omit_idx) == 0:
        print("all datapoints are evaluated!")
        return
    


if __name__ == "__main__":
    main()
