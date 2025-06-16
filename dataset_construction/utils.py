import json
import pandas as pd
from typing import Dict, Any, List


def write_data(jsonl_file: str, task: Dict[str, Any]):
    with open(jsonl_file, 'a') as f:
        f.write(json.dumps(task) + '\n')


def load_needles(file_name: str) -> List[str]:
    """Load the file and return a list of needles"""
    try:
        df = pd.read_csv(file_name, sep=";")
    except FileNotFoundError:
        print(f"File {file_name} not found")

    # Process your needles data here
    pass