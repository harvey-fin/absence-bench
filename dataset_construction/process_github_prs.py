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
from pathlib import Path

# Define the header prefixes that should never be removed
HEADER_PREFIXES = ("+++", "---", "diff --git", "index ", "@@")


def should_delete_line(
    line, is_changed_line, allow_context_deletion, prob_changed, prob_context
):
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


def process_diff_text(
    original_diff, allow_context_deletion, prob_changed, prob_context
):
    """
    Process the diff text line by line, randomly deleting lines as specified.

    A changed line is one that starts with '+' or '-' (but not if it is a header line).
    Header lines are defined in HEADER_PREFIXES.

    In default mode:
      Only changed lines are eligible for deletion.
    In context deletion mode:
      Any non-header line is eligible for deletion, but only omitted changed lines
      (insertions/deletions) are recorded.

    Returns the modified diff text and a list of omitted changed lines.
    """
    modified_lines = []
    # Will hold the exact text of omitted lines that are changed lines.
    omitted_changed_lines = []
    omitted_line_idx = []

    # Split the diff into individual lines.
    diff_lines = original_diff.splitlines()
    unique_lines = [l for l in diff_lines if diff_lines.count(l) == 1]

    for line_idx, line in enumerate(diff_lines):
        # Determine if the line is a "changed" line (starts with '+' or '-').
        # Note: Even if a header line starts with '+' or '-',
        # we mark it as not changed by our criteria;
        # this is handled in should_delete_line via header check.
        is_changed_line = line.startswith("+") or line.startswith("-")

        # if the line has a duplication, then we skip it
        if line not in unique_lines:
            continue

        # Decide whether to delete this line.
        delete_line, track = should_delete_line(
            line, is_changed_line, allow_context_deletion, prob_changed, prob_context
        )

        if delete_line:
            # In the default mode, only changed lines are deleted and tracked.
            if track:
                omitted_changed_lines.append(line)
                omitted_line_idx.append(line_idx)
            # Do not add the line to modified_lines.
        else:
            modified_lines.append(line)

    # Reconstruct the modified diff.
    modified_diff = "\n".join(modified_lines)
    return modified_diff, omitted_changed_lines, omitted_line_idx


def process_github_prs_file(
    input_file, output_file, allow_context_deletion, prob_changed, prob_context
):
    """
    Process a GitHub merged PRs JSON-lines file. For each record (called here an "issue"),
    randomly delete some lines from the 'diff' field.

    For each issue, store:
      - original_diff: the full diff from the input record.
      - modified_diff: the diff after randomly deleting lines.
      - omitted_lines: a list of the exact text of each deleted changed line.
        In the context deletion mode, any deleted context lines are not tracked.
      - Also, include metadata: deletion_mode ("changed_only" or "context_allowed")
        and the probabilities used.

    Other fields are copied from the original record, but the poem-related key is now
    replaced by diff-related keys.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "a", encoding="utf-8"
    ) as f_out:
        for line_num, line in enumerate(f_in):
            try:
                record = json.loads(line)

                # Expect the diff to be under the key "diff".
                if "diff" not in record:
                    print(
                        f"Warning: Line {line_num+1} - No 'diff' field found. Skipping."
                    )
                    continue

                original_diff = record["diff"]
                prob_changed = random.uniform(0.05, 0.5)
                prob_context = prob_changed
                allow_context_deletion = True
                modified_diff, omitted_lines, omitted_idx = process_diff_text(
                    original_diff, allow_context_deletion, prob_changed, prob_context
                )

                # Create the combined entry. We use "issue" terminology.
                combined_entry = {
                    "issue_id": record.get("pr_number", line_num),
                    "original_diff": original_diff,
                    "modified_diff": modified_diff,
                    "omitted_lines": omitted_lines,
                    "omitted_index": omitted_idx,
                    # Add metadata about how deletion was performed:
                    "deletion_metadata": {
                        "deletion_mode": (
                            "context_allowed"
                            if allow_context_deletion
                            else "changed_only"
                        ),
                        "prob_changed": prob_changed,
                        "prob_context": (
                            prob_context if allow_context_deletion else None
                        ),
                    },
                }

                # Copy over other fields from the original record if needed
                # (excluding the old 'diff').
                for key, value in record.items():
                    if key not in ["diff"]:
                        combined_entry[key] = value

                f_out.write(json.dumps(combined_entry) + "\n")

                if (line_num + 1) % 1000 == 0:
                    print(f"Processed {line_num + 1} issues...")
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num+1} - Invalid JSON. Skipping.")
                continue

    print(f"Processing complete. Output saved to {output_file}")


def main():
    """
    The main function
    """
    parser = argparse.ArgumentParser(
        description="Process a GitHub merged PRs JSONL file to produce modified diffs "
        "with random deletion of diff lines (issue version)."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input GitHub merged PRs JSONL file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/github_prs_processed.jsonl",
        help="Path to the output file (default: github_prs_processed.jsonl)",
    )
    parser.add_argument(
        "--allow-context-deletion",
        action="store_true",
        help=(
            "Allow deletion of any non-header line (i.e. context lines included). "
            "By default, only deletion/insertion lines are eligible."
        ),
    )
    parser.add_argument(
        "--prob-changed",
        type=float,
        default=0.1,
        help="Probability of deleting a changed line (default: 0.1)",
    )
    parser.add_argument(
        "--prob-context",
        type=float,
        default=0.05,
        help=(
            "Probability of deleting a context line"
            " (only used if --allow-context-deletion is set; default: 0.05)"
        ),
    )

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output
    allow_context_deletion = args.allow_context_deletion
    prob_changed = args.prob_changed
    prob_context = args.prob_context

    # Check if input file exists.
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist!")
        return

    process_github_prs_file(
        input_file, output_file, allow_context_deletion, prob_changed, prob_context
    )


if __name__ == "__main__":
    main()
