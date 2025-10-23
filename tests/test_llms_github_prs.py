"""
This script tests LLMs on their ability to recall the changes made in a GitHub merge PR diff.
It accepts as input a processed JSON Lines file (e.g. one produced by process_github_prs.py)
containing, for each PR, the following keys:
  - "original_context": The full diff as retrieved from GitHub.
  - "modified_context": The diff after randomly deleting some lines.
  - "omitted_context": A list of the exact text of each changed line
    (insertions/deletions) that was removed.

The LLM’s response is then evaluated by checking if it correctly identifies
    the missing changed lines.
Each evaluation includes the total expected changed lines omitted and the number and list
of those that the model correctly recalled.

Example usage:
    python test_llm_github_prs.py --input_file data/github_prs.jsonl
         [--sample_size N] [--provider_models openai:gpt-4 ...]
         [--output results/prs_gpt-4.json] [--batch_size 5]
"""

import argparse
import concurrent.futures
import json
import random
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Union

from llm_providers import  LLMProvider 


def load_diffs(jsonl_file: str, sample_size: int = None) -> List[Dict[str, Any]]:
    """
    Load diff records from the JSONL file. Optionally sub-sample to a given number.
    Each record is expected to have keys like
        "original_context", "modified_context", and "omitted_context".
    """
    diffs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            diffs.append(json.loads(line))

    if sample_size and sample_size < len(diffs):
        return random.sample(diffs, sample_size)
    return diffs


def evaluate_response(response_list: List[Union[str, int]], 
                      diff_data: Dict[str, Any], use_needle: bool) -> Dict[str, Any]:
    """
    Evaluate the model's response to determine if it correctly identified
        the omitted changed lines.
    The expected omitted changed lines come from the "omitted_lines" list in diff_data.
    The evaluation checks if the response (after cleaning) contains the non-omitted lines and the omitted lines
    """

    original_lines = diff_data["original_context"].split('\n')
    omitted_indices = diff_data["omitted_index"]

    if use_needle:
        needles = diff_data['needles']
    
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
    
    repeat_lines =  list(set([l for l in original_lines if original_lines.count(l) != 1]))
    for line in repeat_lines:
        line_count =  min(response.lower().count("\n"+line.strip().lower()+"\n"), original_lines.count(line))
        results["fp"] += line_count
        for i in range(line_count):
            results["wrongly_identified_lines"].append(line)

    if use_needle:
        for needle in needles:
            if needle.lower() in response.lower():
                results["tp"] += 1
                results["identified_lines"].append(needle)
            else:
                results["fn"] += 1
                results["unidentified_lines"].append(needle)
        
        for idx, line in enumerate(original_lines):
            if line in repeat_lines:
                continue
            # Clean up the line for comparison (remove punctuation, extra spaces, etc.)
            clean_line = line.strip().lower()
            if clean_line and clean_line in response.lower():
                results["fp"] += 1
                results["wrongly_identified_lines"].append(line)
    else:
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
        model_provider: str,
        model_name: str,
        system_prompt: str,
        thinking: bool,
        use_needle: bool,
    ) -> Dict[str, Any]:
    """
    Process a single diff record with the specified model.
    The prompt is constructed using the original and modified diffs from the record.
    The LLM is asked to list the missing
    changed lines (insertions/deletions) from the original diff.
    """
    # Construct the user prompt using the provided data.
    user_message = f"""Here is the complete original diff:

{diff_record['original_context']}

And here is the merge diff after the developer fixed the commit history:

{diff_record['modified_context']}

What changed lines (insertions or deletions) present \
in the original diff are missing in the merge diff (if any)?
List only the missing changed lines, nothing else."""
    
    if use_needle:
            user_message = f"""Here is the complete original diff:
{diff_record['original_context']}

And here is the merge diff after the developer fixed the commit history,\
with some extra liens that is related to Harry Potter novel series

{diff_record['modified_context']}

What lines did I add to the original diff? Please list on only the extra lines, nothing else."""

    try:
        # Retrieve the LLM provider based on model_provider.
        provider = LLMProvider.get_provider(model_provider)

        # Get the model response.
        response = provider.get_response(system_prompt, user_message, model_name, thinking=thinking)

        evaluation = evaluate_response(response, diff_record, use_needle)
        evaluation["id"] = diff_record.get("pr_number", diff_idx)
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
        use_needle: bool = False,
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
    if use_needle:
        system_prompt = (
        "You are helping a software developer determine if their merge"
        " of a pull request was successful. "
        "The developer had to edit the commit history and accidently added some random lines related to Harry Potter characters"
        "They will list the changed lines. "
        "Your job is to figure out if they have added any"
        "insertions from the original merge. "
        "Only pay attention to the insertions."
        )


    for batch_start in tqdm(range(0, total_diffs, batch_size)):
        batch_end = min(batch_start + batch_size, total_diffs)
        current_batch = diffs[batch_start:batch_end]

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for i, diff_record in enumerate(current_batch):
                diff_idx = batch_start + i
                future = executor.submit(
                    process_diff,
                    diff_record,
                    diff_idx,
                    model_provider,
                    model_name,
                    system_prompt,
                    thinking,
                    use_needle,
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        if batch_end < total_diffs:
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
    parser.add_argument("--input_file", type=str, default="data/github_prs.jsonl",
                        help="Path to the processed GitHub PR diffs JSONL file")
    parser.add_argument("--sample_size", type=int,
                        help="Number of diff records to sample for testing (default: use all)")
    parser.add_argument("--provider_models", type=str, nargs="+", default=["openai:gpt-4o"],
                        help=('Provider and model pairs in the format "provider:model"'
                            ' (e.g., "openai:gpt-4o anthropic:claude-3-opus")'))
    parser.add_argument("--output", type=str,
                        help="Path to save the test results")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of API calls to batch together (default: 5)")
    parser.add_argument("--thinking", action='store_true',
                        help="Whether to enable the thinking mode or not")
    parser.add_argument("--use_needle", action="store_true",
                        help='evalute with the NIAH setting')

    args = parser.parse_args()

    diffs_path = Path(args.input_file)
    if not diffs_path.exists():
        print(f"Error: Diffs file '{args.input_file}' does not exist!")
        return

    diffs = load_diffs(args.input_file, args.sample_size)
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

        print(f"Testing provider: {provider}, model: {model}")

        try:
            results = test_model(diffs, provider, model, 
                                 args.batch_size, args.thinking, args.use_needle)
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


if __name__ == "__main__":
    main()
