"""
This consumer script processes a GitHub merged PR JSON Lines file
(e.g. one produced by github_merged_prs.py)
and produces an output file where, for each issue (i.e. each PR record),
the diff has been modified by randomly deleting some lines.
In the output JSON for each PR, both the original diff
and the modified diff are stored, along with a list
of the exact text of each omitted changed line.
By default, only lines with changes (insertions or deletions) are eligible for removal.
With the command‐line flag --allow-context-deletion, any non‐header line can be deleted,
with separate probabilities for deletion of changed lines and context lines
(though only omitted changed lines are tracked).

Usage:
    python process_github_prs.py input.jsonl [-o OUTPUT] [--allow-context-deletion]
         [--prob-changed PROB] [--prob-context PROB]

"""

import argparse
import json
import random
from typing import List
from pathlib import Path
from utils import write_data, load_needles
from tqdm import tqdm

# Define the header prefixes that should never be removed
HEADER_PREFIXES = ("+++", "---", "diff --git", "index ", "@@")


def should_delete_line(line: str, is_changed_line: bool, 
                       allow_context_deletion: bool, prob_changed: float,
                       prob_context: float):
    """
    Decide whether to delete a line based on the following:

    In default mode (allow_context_deletion is False):
      - Only lines that are "changed" (i.e. they start with '+' or '-')
         are eligible for deletion.
      - The probability of deletion is prob_changed.

    In context deletion mode (allow_context_deletion is True):
      - For changed lines the deletion probability is prob_changed.
      - For non-header context lines
        (which do not start with '+' or '-' and are not headers),
        the deletion probability is prob_context.
    
    Inputs:
        line: one line of diff in a PR record
        is_changed_line: whether the line is a changed line
        allow_context_deletion: use default/context deletion mode
        prob_changed: probability that a changed line is omitted
        prob_context: probability that a non-changed line is omitted

    Returns a tuple (delete_line, track_deletion) where:
      - delete_line: Boolean indicating whether the line should be removed.
      - track_deletion: Boolean indicating whether this deletion should be
                        tracked in the omitted list.
                        (In context-mode, deletions of context lines are not tracked.)
    """
    # Always keep header lines: headers are lines that start with any of the header prefixes.
    if line.startswith(HEADER_PREFIXES):
        return (False, False)

    # Default mode: Only changed lines are considered
    if not allow_context_deletion:
        if is_changed_line:
            if random.random() < prob_changed:
                return (True, True)
        return (False, False)

    # In context deletion mode, all non-header lines are eligible.
    if is_changed_line:
        if random.random() < prob_changed:
            return (True, True)  # Track deletion for changed lines.
    else:
        # For context lines, use the prob_context threshold, but do not track them.
        if random.random() < prob_context:
            return (True, False)
    return (False, False)


def process_diff_text(original_diff: str, allow_context_deletion: bool, 
                      prob_changed: float, prob_context: float, 
                      needles: List[str], placeholder: str):
    """
    Process the diff text line by line, randomly deleting lines as specified.

    A changed line is one that starts with '+' or '-' (but not if it is a header line).
    Header lines are defined in HEADER_PREFIXES.

    In default mode:
      Only changed lines are eligible for deletion.
    In context deletion mode:
      Any non-header line is eligible for deletion, but only omitted changed lines
      (insertions/deletions) are recorded.
    
    Inputs:
        original_diff: the complete diff of a PR record
        allow_context_deletion: use default/context deletion mode
        prob_changed: probability that a changed line is omitted
        prob_context: probability that a non-changed line is omitted
        needles: a list of needles that are inserted into omitted indices
        placeholder: string to indicate omissions

    Returns the modified diff text, a list of omitted changed lines,
            and a list of used_needles [optional]
    """
    modified_lines = []
    # Will hold the exact text of omitted lines that are changed lines.
    omitted_changed_lines = []
    omitted_line_idx = []

    # Keep track of needles that are inserted into the modified context if any
    used_needles = []
        
    # Split the diff into individual lines.
    diff_lines = original_diff.splitlines()
    unique_lines = [l for l in diff_lines if diff_lines.count(l) == 1]

    for line_idx, line in enumerate(diff_lines):
        # Determine if the line is a "changed" line (starts with '+' or '-').
        # Note: Even if a header line starts with '+' or '-',
        # we mark it as not changed by our criteria;
        # this is handled in should_delete_line via header check.
        is_changed_line = line.startswith("+") or line.startswith("-")

        # If the line has a duplication, then we skip it
        if line not in unique_lines:
            continue

        # Decide whether to delete this line.
        delete_line, track = should_delete_line(
            line, 
            is_changed_line, 
            allow_context_deletion, 
            prob_changed, 
            prob_context
        )

        # keep track of used needles if any

        if delete_line:
            # In the default mode, only changed lines are deleted and tracked.
            if track:
                omitted_changed_lines.append(line)
                omitted_line_idx.append(line_idx)

            # In the NIAH test setting, add the needle to the modified lines
            if needles:
                remain_needles = [n for n in needles if n not in used_needles]
                if remain_needles:
                    needle = random.choice(remain_needles)
                    modified_lines.append(needle)
                    used_needles.append(needle)
            # If using placeholders, then add the placeholder instead
            elif placeholder:
                modified_lines.append(placeholder)
        else:
            modified_lines.append(line)

    # Reconstruct the modified diff.
    modified_diff = "\n".join(modified_lines)
    return modified_diff, omitted_changed_lines, omitted_line_idx, used_needles


def process_github_prs_file(input_file: str, allow_context_deletion: bool, 
                            prob_changed: float, prob_context: float, 
                            use_needle: bool, use_placeholders: bool):
    """
    Process a GitHub merged PRs JSON-lines file. For each record (called here an "issue"),
    randomly delete some lines from the 'diff' field.

    For each issue, store:
      - original_context: the full diff from the input record.
      - modified_context: the diff after randomly deleting lines.
      - omitted_context: a list of the exact text of each deleted changed line.
        In the context deletion mode, any deleted context lines are not tracked.
      - Also, include metadata: deletion_mode ("changed_only" or "context_allowed")
        and the probabilities used.

    Note:
        use_needle: whether to test with the NIAH setting
                     replace PATH_TO_NEEDLES_FILE with the path to needles file
        use_placeholders: whether to indicate omission with placeholders
                          default placeholder: <missing line>

    Other fields are copied from the original record, but the poem-related key is now
    replaced by diff-related keys.
    """
    input_path = Path(input_file)

    # If testing NIAH setting, load needles from file
    if use_needle:
        print(f"Testing the NIAH setting under GitHub PRs domain")
        needles_file = "PATH_TO_NEEDLES_FILE"
        assert needles_file != "PATH_TO_NEEDLES_FILE",\
            "Need to specify the path to needles file!"

        needles = load_needles(needles_file)
    else:
        needles = None

    # If using placeholders, specify the placeholder here
    if use_placeholders:
        print(f"Testing placeholder. Default: <missing lines>")
        placeholder = "<missing line>"
    else:
        placeholder = None

    print("===== Processing GitHub PRs Data =====")
    
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_num, line in tqdm(enumerate(f_in)):
            try:
                record = json.loads(line)

                # Expect the diff to be under the key "diff".
                if "diff" not in record:
                    print(
                        f"Warning: Line {line_num+1} - No 'diff' field found. Skipping."
                    )
                    continue

                original_diff = record["diff"]

                modified_diff, omitted_lines, omitted_idx, used_needles = process_diff_text(
                    original_diff, 
                    allow_context_deletion, 
                    prob_changed, 
                    prob_context,
                    needles,
                    placeholder
                )

                # Create the combined entry. We use "issue" terminology.
                combined_entry = {
                    "id": record.get('id', line_num),
                    "original_context": original_diff,
                    "modified_context": modified_diff,
                    "omitted_context": omitted_lines,
                    "omitted_index": omitted_idx,
                    # Add metadata about how deletion was performed:
                    "metadata": {
                        "pr_number": record.get("pr_number", line_num),
                        "repo": record.get("repo", line_num),
                        "title": record.get("title", line_num),
                        "author": record.get("author", line_num),
                        "additions": record.get("additions", line_num),
                        "deletions": record.get("deletions", line_num),
                        "total_changes": record.get("total_changes", line_num),
                        "html_url": record.get("html_url", line_num),
                        "omission_probability": prob_changed,
                    },
                }

                # Copy over other fields from the original record if needed
                # (excluding the old 'diff').
                if use_needle:
                    combined_entry["needles"] = used_needles
                    write_data(f'data/github_prs_needles.jsonl', combined_entry)
                elif use_placeholders:
                    combined_entry["metadata"]["placeholder"] = placeholder
                    write_data(f'data/github_prs_placeholder.jsonl', combined_entry)
                else:
                    write_data(f'data/github_prs.jsonl', combined_entry)

            except json.JSONDecodeError:
                print(f"Warning: Line {line_num+1} - Invalid JSON. Skipping.")
                continue

    print(f"Processing complete. Output saved under data/")


def main():
    """
    The main function
    """
    parser = argparse.ArgumentParser(
        description="Process a GitHub merged PRs JSONL file to produce modified diffs "
        "with random deletion of diff lines (issue version)."
    )
    parser.add_argument("--input_file", type=str, 
                        help="Path to the input GitHub merged PRs JSONL file")
    parser.add_argument("--allow-context-deletion", action="store_true",
                        help=("Allow deletion of any non-header line "
                        "(i.e. context lines included). "
                        "By default, only deletion/insertion lines are eligible."))
    parser.add_argument("--prob-changed", type=float, default=0.1,
                        help="Probability of deleting a changed line (default: 0.1)")
    parser.add_argument("--prob-context", type=float, default=0,
                        help=("Probability of deleting a context line"
                        " (only used if --allow-context-deletion is set; default: 0)"))
    parser.add_argument('--use_needle', action="store_true",
                        help='experiment with needle in a haystack')
    parser.add_argument('--use_placeholders', action="store_true",
                        help='use placeholders to help identify omissions')
    parser.add_argument('--random_seed', type=int, default=42,
                        help="random seed")
    args = parser.parse_args()

    random.seed(args.random_seed)

    input_file = args.input_file
    allow_context_deletion = args.allow_context_deletion
    prob_changed = args.prob_changed
    prob_context = args.prob_context
    use_needle = args.use_needle
    use_placeholders = args.use_placeholders

    # Check if input file exists.
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist!")
        return

    process_github_prs_file(
        input_file, 
        allow_context_deletion, 
        prob_changed, 
        prob_context,
        use_needle,
        use_placeholders
    )


if __name__ == "__main__":
    main()
