"""
This script processes a Poetry JSON Lines file
and produces an output file where, each poem
 has been modified by randomly deleting some lines.
In the output JSON for each poem, both the original poem
and the modified poem are stored, along with a list
of the exact text of each omitted changed line.

Usage:
    python process_github_prs.py input.jsonl [-o OUTPUT] [--allow-context-deletion]
         [--prob-changed PROB] [--prob-context PROB]

"""
import json
import argparse
import random
from pathlib import Path
from utils import load_needles, write_data
from tqdm import tqdm


def process_poetry_file(input_file: str, omission_prob: float = 0.1, 
                        max_lines = None, use_needle: bool = False, 
                        use_placeholders: bool = True):
    """
    Process a poetry JSONL file to create a dataset with each poem having
    both original and modified versions in the same JSON object
    """

    # If testing NIAH setting, load needles from file
    if use_needle:
        print(f"Testing the NIAH setting under poetry domain")
        needles_file = "PATH_TO_NEEDLES_FILE"
        assert needles_file != "PATH_TO_NEEDLES_FILE",\
            "Need to specify the path to needles file!"

        needles = load_needles(needles_file)
    else:
        needles = None

    if use_placeholders:
        print(f"Testing placeholder. Default: <missing lines>")
        placeholder = "<missing line>"
    else:
        placeholder = None

    print("===== Processing Poetry Data =====")

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in tqdm(enumerate(f_in)):
            try:
                # Parse the JSON object
                poem_data = json.loads(line)
                
                # Check if the poem content is in 'text' or 'content' field
                poem_field = 'text' if 'text' in poem_data else 'content'
                if poem_field not in poem_data:
                    print(f"Warning: Line {line_num+1} - Could not find poem content. Skipping.")
                    continue

                # Keep track of used needles if any
                used_needles = []
                
                # Get the original poem text
                original_poem = poem_data[poem_field]
                
                # Split the poem into lines
                poem_lines = original_poem.split('\n')
                
                # Apply truncation if max_lines is specified
                if max_lines is not None:
                    poem_lines = poem_lines[:max_lines]
                    # Reconstruct the truncated original poem
                    original_poem = '\n'.join(poem_lines)
                
                # Create the modified version with some lines omitted
                modified_lines = []
                omitted_poem = []
                omitted_indices = []  # Track indices of omitted lines

                for i, poem_line in enumerate(poem_lines):
                    # Keep the line with probability (1-p)
                    if random.random() > omission_prob:
                        modified_lines.append(poem_line)
                    else:
                        if use_needle:
                            modified_lines.append(poem_line)
                            remain_needles = [n for n in needles if n not in used_needles]
                            if remain_needles:
                                needle = random.choice(remain_needles)
                                modified_lines.append(needle)
                                used_needles.append(needle)
                            else:
                                needle = ""
                        elif placeholder:
                            modified_lines.append(placeholder)

                        omitted_poem.append(poem_line)
                        omitted_indices.append(i)  # Store the index of the omitted line
                
                # Create the modified poem
                modified_poem = '\n'.join(modified_lines)
                
                # Create a combined entry with both versions
                combined_entry = {
                    'id': poem_data.get('id', line_num),
                    'original_context': original_poem,
                    'modified_context': modified_poem,
                    'omitted_context': omitted_poem,
                    'omitted_index': omitted_indices,  # Add the omitted line indices
                    "metadata": {
                        'poem_length': len(poem_lines),
                        'omission_probability': omission_prob,
                        'n_omitted': len(omitted_indices)
                    },
                }
                
                # Copy any other fields from the original data if needed
                for key, value in poem_data.items():
                    if key != poem_field and key != 'id' and key not in combined_entry:
                        combined_entry["metadata"][key] = value
                
                # Write the combined entry to the output file
                if use_needle:
                    combined_entry['needles'] = used_needles
                    write_data(f'data/poetry_needles.jsonl', combined_entry)
                elif use_placeholders:
                    combined_entry['placeholder'] = placeholder
                    write_data(f'data/poetry_placeholder.jsonl', combined_entry)
                else:
                    write_data(f'data/poetry.jsonl', combined_entry)
                    
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num+1} - Invalid JSON. Skipping.")
                continue
            except Exception as e:
                import pdb; pdb.set_trace()
                print(f"Error processing line {line_num+1}: {e}")
                continue
                
    print(f"Process complete. Output saved under data/")


def main():
    parser = argparse.ArgumentParser(
        description='Process poetry dataset to create original and modified versions.'
    )
    parser.add_argument('--input_file', type=str, default="data/poetry_raw.jsonl",
                        help='Path to the input poetry.jsonl file')
    parser.add_argument('-p', '--prob', type=float, default=0.1,
                        help='Probability of omitting a line (default: 0.1)')
    parser.add_argument('-m', '--max_lines', type=int,
                        help='Maximum number of lines in the poem')
    parser.add_argument('--use_needle', action="store_true",
                        help='experiment with needle in a haystack')
    parser.add_argument('--use_placeholders', action="store_true",
                        help='use placeholders to help identify omissions')
    parser.add_argument('--random_seed', type=int, default=42,
                        help="random seed")

    args = parser.parse_args()
    random.seed(args.random_seed)
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist!")
        return
    
    # Process the poetry file
    process_poetry_file(
        args.input_file, 
        args.prob, 
        args.max_lines, 
        args.use_needle,
        args.use_placeholders)


if __name__ == "__main__":
    main()