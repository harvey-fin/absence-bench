import subprocess
from typing import *
import os    
import random

"""
collect the pull requests of top twenty repos with most prs
"""

def collect_pull_requests():
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
        prob_changed = str(round(random.uniform(0.1, 0.3),3))
        prob_context = str(round(random.uniform(0.05, 0.1),3))
        infile_path = os.path.join(directory_path, file)
        subprocess.run(["python", "dataset_construction/process_github_prs.py", 
                        infile_path, 
                        "--output", "data/github_prs_analysis.jsonl",
                        "--prob-changed", prob_changed, 
                        "--prob-context", prob_context, 
                        "--allow-context-deletion"])

if __name__ == "__main__":
    process_pr_data("data/merges/")
