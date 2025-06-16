"""
This script run the data process script for all three domains:
poetry: process_poetry.py
numerical sequences: generate_numeric.py
github_prs: github_merged_prs; process_github_prs.py
with a specific random seed

Usage:
    python generate_data.py --random_seed 42
"""

import subprocess
import os    
import random


def collect_pull_requests():
    """collect the pull requests of top twenty repos with most prs"""

    repo_list = ["godotengine/godot", "nodejs/node", "zed-industries/zed", "ggml-org/llama.cpp", "php/php-src", "rust-lang/rust", \
                    "laravel/framework", "helix-editor/helix", "facebook/react", "torvalds/linux", "vercel/next.js", "vuejs/core", \
                        "microsoft/TypeScript", "neovim/neovim", "facebook/react-native", "prettier/prettier", "electron/electron", \
                            "angular/angular", "helm/helm"]
    for repo in repo_list:
        subprocess.run(["python", "dataset_construction/github_merged_prs.py", repo], capture_output=True, text=True)


def process_pr_data(directory_path: str):
    try:
        files = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
    
    for file in files:
        # the default branch only includes prob_changed = 0.1
        prob_changed = 0.1
        prob_context = 0
        infile_path = os.path.join(directory_path, file)

        # pass --use_needle here to generate data under the NIAH setting; --use_placeholders to generate data with placeholders
        subprocess.run(["python", "dataset_construction/process_github_prs.py", 
                        "--input_file", infile_path, 
                        "--prob-changed", prob_changed,
                        "--prob-context", prob_context]
                      )

if __name__ == "__main__":
    collect_pull_requests()
    process_pr_data("data/merges/")
