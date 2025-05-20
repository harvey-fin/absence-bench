"""
Download closed and merged pull requests of a given GitHub repository
from the last year. For each PR, download the diff (with context as given by GitHub)
and also generate a diff that includes only the changed lines (i.e. without unchanged context).
Count the number of additions and deletions, and only include PRs where the total number of
changed lines is within a specified range (min and max).
Stop as soon as a given quota of PRs have been processed.
Output each approved merge as a JSON-lines file with relevant metadata.

Usage:
    python github_merged_prs.py owner/repo [--min-lines MIN] [--max-lines MAX] [--quota N]

"""

import argparse
import datetime
import json
import os
import sys
import time

import requests

# Configuration
GITHUB_API_TOKEN = os.environ.get("GITHUB_API_KEY")
REQUEST_SLEEP_SECONDS = 1  # to avoid triggering GitHub rate limits


def get_github_headers(accept_type="application/vnd.github.v3+json"):
    """
    Returns headers for the GitHub API request.
    """
    headers = {"Accept": accept_type}
    if GITHUB_API_TOKEN:
        headers["Authorization"] = f"token {GITHUB_API_TOKEN}"
    return headers


def get_date_one_year_ago():
    """
    Returns a datetime object for one year ago from now (UTC).
    """
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=365)


def fetch_merged_prs(repo, page=1, per_page=100):
    """
    Fetch PRs from the given repository (closed PRs) for one page.
    We will later check for merged_at and merged date.
    """
    url = f"https://api.github.com/repos/{repo}/pulls"
    params = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "page": page,
        "per_page": per_page,
    }
    headers = get_github_headers()
    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code != 200:
        print("Error fetching PRs:", response.status_code, response.text)
        sys.exit(1)
    return response.json()


def fetch_diff_for_pr(repo, pr_number):
    """
    Fetch the PR’s diff (with context) using the GitHub API.
    This function requests the diff version by setting the Accept header.
    """
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    # This Accept header returns the diff format.
    headers = get_github_headers(accept_type="application/vnd.github.v3.diff")
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        print(
            f"Error fetching diff for PR #{pr_number}:",
            response.status_code,
            response.text,
        )
        return None
    return response.text


def count_line_changes(diff_text):
    """
    Count the number of additions and deletions in the diff.
    We ignore metadata lines: only count lines starting with
        '+' or '-' except file header lines ("+++" or "---").
    Returns a tuple (additions, deletions)
    """
    additions = 0
    deletions = 0
    for line in diff_text.splitlines():
        # Skip file headers
        if (
            line.startswith("+++")
            or line.startswith("---")
            or line.startswith("diff --git")
            or line.startswith("index ")
            or line.startswith("@@")
        ):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def main():
    """
    The main function
    """
    parser = argparse.ArgumentParser(
        description="Download closed & merged PRs with diff details"
        " from a GitHub repository."
    )
    parser.add_argument(
        "repository", type=str, help="Full name of the repository (e.g., owner/repo)"
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=10,
        help=(
            "Minimum total number of added+deleted lines for"
            " a PR to be included (default: 0)"
        ),
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help=(
            "Maximum total number of added+deleted lines for"
            " a PR to be included (default: 1000)"
        ),
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=50,
        help="Number of PR merges to save (default: 10)",
    )

    args = parser.parse_args()

    repo = args.repository
    min_lines = args.min_lines
    max_lines = args.max_lines
    quota = args.quota

    cutoff_date = get_date_one_year_ago()

    print(f"Processing repository {repo}")
    print(
        f"Only PRs merged after {cutoff_date.isoformat()} and with "
        f"line changes in [{min_lines}, {max_lines}] will be kept"
    )
    print(f"Looking for up to {quota} merges.")

    collected_prs = []
    page = 1
    per_page = 100

    while len(collected_prs) < quota:
        prs = fetch_merged_prs(repo, page=page, per_page=per_page)
        if not prs:
            print("No more PRs found.")
            break

        for pr in prs:
            # Check that the PR is merged and that merged_at is not None.
            merged_at_str = pr.get("merged_at")
            if not merged_at_str:
                continue
            merged_at = datetime.datetime.strptime(
                merged_at_str, "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=datetime.timezone.utc)

            # Skip PRs merged before our cutoff
            if merged_at < cutoff_date:
                continue

            # Get the diff for the PR.
            pr_number = pr.get("number")
            print(f"Processing PR #{pr_number} (merged at {merged_at_str})")
            diff_text = fetch_diff_for_pr(repo, pr_number)
            if diff_text is None:
                continue

            additions, deletions = count_line_changes(diff_text)
            total_changes = additions + deletions
            print(
                f"PR #{pr_number} - +{additions} / -{deletions} (total {total_changes} changes)"
            )

            # Filter by total changes.
            if total_changes < min_lines or total_changes > max_lines:
                print("\tDoes not meet line change criteria. Skipping.")
                continue

            # Prepare a record with required metadata.
            record = {
                "repo": repo,
                "pr_number": pr_number,
                "title": pr.get("title"),
                "merged_at": merged_at_str,
                "author": pr.get("user", {}).get("login"),
                "additions": additions,
                "deletions": deletions,
                "total_changes": total_changes,
                "diff": diff_text,
                "html_url": pr.get("html_url"),
            }
            collected_prs.append(record)
            if len(collected_prs) >= quota:
                break

            # Sleep briefly to avoid hitting rate limits
            time.sleep(REQUEST_SLEEP_SECONDS)

        page += 1
        # A delay between page requests.
        time.sleep(REQUEST_SLEEP_SECONDS)

    print(f"\nCollected {len(collected_prs)} merged PRs. Saving output.")

    # Save output as JSON Lines file.
    today = datetime.datetime.now().date().isoformat()
    outfile_name = f"{repo.replace('/', '__')}-{today}-merged-prs.jsonl"
    output_dir = os.path.join("data", "merges")
    os.makedirs(output_dir, exist_ok=True)  # Create a folder for data output if needed.
    outfile_path = os.path.join(output_dir, outfile_name)

    with open(outfile_path, "w", encoding="utf-8") as outfile:
        for record in collected_prs:
            json.dump(record, outfile)
            outfile.write("\n")

    print(f"Output saved to {outfile_path}")


if __name__ == "__main__":
    main()
