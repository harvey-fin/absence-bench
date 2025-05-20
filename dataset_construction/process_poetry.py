import json
import argparse
import random
from pathlib import Path
from generate_needles import load_data

def process_poetry_file(input_file, output_file, omission_prob=0.1, max_lines=None, use_needles:bool=False, use_placeholders:bool=False):
    """
    Process a poetry JSONL file to create a dataset with each poem having
    both original and modified versions in the same JSON object
    """
    # test different omission_prob

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in):
            try:
                # Parse the JSON object
                poem_data = json.loads(line)
                
                # Check if the poem content is in 'text' or 'content' field
                poem_field = 'text' if 'text' in poem_data else 'content'
                if poem_field not in poem_data:
                    print(f"Warning: Line {line_num+1} - Could not find poem content. Skipping.")
                    continue

                if use_needles:
                    needles = load_data("data/harrypotter/Characters.csv")
                used_needles = []
                
                # Get the original poem text
                original_poem = poem_data[poem_field]
                
                # Split the poem into lines
                poem_lines = original_poem.split('\n')

                # to test for length_to_accs, make max_lines uniformly distributed over (100, 1000)
                max_lines = int(random.random()*900)+100
                
                # Apply truncation if max_lines is specified
                if max_lines is not None:
                    poem_lines = poem_lines[:max_lines]
                    # Reconstruct the truncated original poem
                    original_poem = '\n'.join(poem_lines)
                
                # Create the modified version with some lines omitted
                modified_lines = []
                omitted_indices = []  # Track indices of omitted lines
                
                # randomly set the omission prob
                # omission_prob = random.uniform(0.05, 0.5)
                omission_prob = 0.1

                for i, poem_line in enumerate(poem_lines):
                    # Keep the line with probability (1-p)
                    if random.random() > omission_prob:
                        modified_lines.append(poem_line)
                    else:
                        if use_needles:
                            modified_lines.append(poem_line)
                            remain_needles = [n for n in needles if n not in used_needles]
                            if remain_needles:
                                needle = random.choice(remain_needles)
                            else:
                                needle = ""
                            if needle:
                                modified_lines.append(needle)
                                used_needles.append(needle)
                        omitted_indices.append(i)  # Store the index of the omitted line
                # If we've removed all lines, keep at least one random line
                if not modified_lines and poem_lines:
                    random_index = random.randrange(len(poem_lines))
                    modified_lines = [poem_lines[random_index]]
                    # Update omitted_indices to reflect that we're keeping this line
                    if random_index in omitted_indices:
                        omitted_indices.remove(random_index)
                
                # Create the modified poem
                modified_poem = '\n'.join(modified_lines)
                
                # Create a combined entry with both versions
                combined_entry = {
                    'id': poem_data.get('id', line_num),
                    'original_poem': original_poem,
                    'modified_poem': modified_poem,
                    'omission_probability': omission_prob,
                    'omitted_line_indices': omitted_indices  # Add the omitted line indices
                }
                if use_needles:
                    combined_entry['needles'] = used_needles
                
                # Copy any other fields from the original data if needed
                for key, value in poem_data.items():
                    if key != poem_field and key != 'id' and key not in combined_entry:
                        combined_entry[key] = value
                
                # Write the combined entry to the output file
                f_out.write(json.dumps(combined_entry) + '\n')
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} poems...")
                    
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num+1} - Invalid JSON. Skipping.")
                continue
            except Exception as e:
                import pdb; pdb.set_trace()
                print(f"Error processing line {line_num+1}: {e}")
                continue
                
    print(f"Processing complete. Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process poetry dataset to create original and modified versions.')
    parser.add_argument('--input_file', type=str, help='Path to the input poetry.jsonl file')
    parser.add_argument('-o', '--output', type=str, default='poetry_processed.jsonl', 
                        help='Path to the output file (default: poetry_processed.jsonl)')
    parser.add_argument('-p', '--prob', type=float, default=0.1,
                        help='Probability of omitting a line (default: 0.1)')
    parser.add_argument('-m', '--max_lines', type=int,
                        help='Maximum number of lines in the poem')
    parser.add_argument('--use_needles', action="store_true",
                        help='experiment with needle in a haystack')
    parser.add_argument('--use_placeholders', action="store_true",
                        help='use placeholders to help identify omissions')

    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist!")
        return
    
    if args.use_placeholders:
        generate_placeholder(args.output)
        return
    
    if args.use_needles:
        generate_needles(args.output)
        return
    
    # Process the poetry file
    process_poetry_file(args.input_file, args.output, args.prob, args.max_lines, args.use_needles)

def generate_needles(output_file: str):
    """open the file and insert a placeholder to all omitted indexes"""
    needles = load_data("data/harrypotter/Characters.csv") 
    used_needles = []

    input_file = "data/poetry_default.jsonl"
    poems = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            poems.append(json.loads(line))
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for p in poems:
            line_list = p['original_poem'].split('\n')
            new_poem = {}
            new_poem['modified_poem'] = []
            new_poem['original_poem'] = p['original_poem']
            for idx, line in enumerate(line_list):
                new_poem['modified_poem'].append(line)
                if idx in p['omitted_line_indices']:
                    remain_needles = [n for n in needles if n not in used_needles]
                    if remain_needles:
                        needle = random.choice(needles)
                    else:
                        needle = ""
                    if needle:
                        new_poem['modified_poem'].append(needle)
                        used_needles.append(needle)

            for key, value in p.items():
                if key not in new_poem.keys():
                    new_poem[key] = value

            f_out.write(json.dumps(new_poem) + '\n') 


def generate_placeholder(output_file: str):
    """open the file and insert a placeholder to all omitted indexes"""
    placeholder = "_" * 10

    input_file = "data/poetry_default.jsonl"
    poems = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            poems.append(json.loads(line))
    
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for p in poems:
            line_list = p['original_poem'].split('\n')
            new_poem = {}
            new_poem['original_poem'] = p['original_poem']
            for idx in p["omitted_line_indices"]:
                line_list[idx] = placeholder 
            new_poem['modified_poem'] = "\n".join(line_list)
            for key, value in p.items():
                if key not in new_poem.keys():
                    new_poem[key] = value

            f_out.write(json.dumps(new_poem) + '\n') 


if __name__ == "__main__":
    main()