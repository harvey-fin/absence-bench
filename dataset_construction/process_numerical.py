'''
This file contains all the code to generate datasets of numerical tasks for absence-bench. 

Following the setup of the rest of the repo, datasets will be generated as jsonl files.
jsonl files are a simple format to store json objects in a file, one object per line.
'''

import numpy as np
import random
import argparse
from tqdm import tqdm
from typing import Dict, Any
from utils import write_data


def generate_numerical_task(n_numbers: int, omission_prob: float, 
                            min_num: int = 0, step_size: int = 1, 
                            order: str = 'ascending',
                            use_placeholders: bool = False, random_seed: int = 42) -> Dict[str, Any]:
    '''
    Generates a numerical task with the given parameters.
    A numerical task is a dictionary with the original sequence, the sequence with some numbers omitted, and the metadata.

    Inputs:
        n_numbers: amount of numbers in the sequence
        omission_prob: the probability of omitting a number from the sequence
        min_num: the minimum number in the sequence
        step_size: The step size between each number. Relevant for arithmetic, squares, cubes sequences.
        order: The order of the sequence. "ascending", "descending", "random"
    
    Output:
        original_sequence: The original sequence
        user_sequence: The sequence with some numbers omitted
        omitted_mask: the mask of the numbers that were omitted from the sequence
        metadata:
            task_type: will always be "numerical"
            min_num: the minimum number in the sequence
            step_size: the step size between each number
            order: the order of the sequence
            n_omitted: the numbers that were omitted from the sequence
    '''

    result = {}

    if use_placeholders:
        print(f"Testing placeholder. Default: <missing lines>")
        placeholder = "<missing line>"
    else:
        placeholder = None

    base_sequence = np.arange(min_num, min_num + n_numbers * step_size, step_size)
    assert len(base_sequence) == n_numbers, "Base sequence length does not match n_numbers"
    
    # for the metadata, we need the original sequence in numeric form to get min, max, etc.
    base_sequence_numeric = base_sequence.copy()

    if order == 'ascending' or order == 'increasing' or order == 'asc':
        base_sequence = np.sort(base_sequence)
    elif order == 'descending' or order == 'decreasing' or order == 'desc':
        base_sequence = np.sort(base_sequence)[::-1]
    elif order == 'random':
        base_sequence = np.random.permutation(base_sequence)
    
    base_sequence = base_sequence.astype(int)

    # omitted_mask is a boolean array that is True for the numbers that are not omitted
    # ensure that we have at least one number omitted and not every number omitted
    omitted_mask = np.random.binomial(1, 1-omission_prob, len(base_sequence)).astype(bool)
    while np.sum(omitted_mask) == 0 or np.sum(omitted_mask) == len(base_sequence):
        omitted_mask = np.random.binomial(1, 1-omission_prob, len(base_sequence)).astype(bool)
    n_omitted = len(omitted_mask) - np.sum(omitted_mask)
    omitted_indices = np.where(~np.array(omitted_mask))[0]

    # Create user sequence by removing omitted numbers from base sequence
    if use_placeholders:
        user_list = []
        for idx, num in base_sequence:
            if idx in omitted_indices:
                user_list.append(placeholder)
            else:
                user_list.append(str(num))
        modified_sequence = "\n".join(user_list)

    else:
        user_sequence = np.array([x for x, mask in zip(base_sequence, omitted_mask) if mask])
        modified_sequence = "\n".join([str(x) for x in user_sequence.tolist()])
        
    # collect metadata
    min_num = int(np.min(base_sequence_numeric))
    max_num = int(np.max(base_sequence_numeric))
    order = order
    n_omitted = int(n_omitted)
    result['original_context'] = "\n".join([str(x) for x in base_sequence.tolist()])
    result['modified_context'] = modified_sequence
    result['omitted_context'] = [str(x) for x, mask in zip(base_sequence.tolist(), omitted_mask) if not mask]
    result['omitted_index'] = [int(i) for i in omitted_indices]
    result['metadata'] = {
        'n_numbers': n_numbers,
        'omission_probability': omission_prob,
        'n_omitted': n_omitted,
        'min_num': min_num,
        'max_num': max_num,
        'step_size': step_size,
        'order': order,
    }

    return result


def generate_numerical_dataset():
    '''
    Generates an entire dataset of numerical tasks.
    Currently only creates arithmetic sequence datasets. 
    
    by default, the dataset would contain less long-sequence data, use only [ascending] order and [base10] system
    '''
    parser = argparse.ArgumentParser(
        description='Process numerical sequences to create original and modified versions.'
    )
    parser.add_argument('--use_placeholders', action="store_true",
                        help='use placeholders to help identify omissions')
    parser.add_argument('--random_seed', type=int, default=42,
                        help="random seed")
    
    args = parser.parse_args()

    # use random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # generate arithmetic sequence type data
    omission_probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    step_sizes = [1, 2, 4, 7, 13]
    orders = ['ascending', 'descending', 'random']
    n_numbers = [50, 100, 500, 750, 1000, 1200]
    # set the distribution of numbers within each bin
    number_dist = [240, 360, 240, 240, 120]
    id = 0 

    print("===== Processing Numerical Data =====")

    # generate a total of 1200 datapoints
    for i in tqdm(range(len(n_numbers)-1)):
        for _ in range(number_dist[i]):
            n = random.randint(n_numbers[i], n_numbers[i+1])
            omission_prob = random.choice(omission_probs)
            step_size = random.choice(step_sizes)
            order = random.choice(orders)
            # only use base10 since the other two would make the sequence 2x-3x longer
            min_num = random.randint(0, 10000)
            
            if args.use_placeholders:
                curr_task = generate_numerical_task(
                    n_numbers=n, 
                    omission_prob=omission_prob,
                    min_num=min_num, 
                    step_size=step_size, 
                    order=order,
                    use_placeholders=True
                )
            else:
                 curr_task = generate_numerical_task(
                    n_numbers=n, 
                    omission_prob=omission_prob,
                    min_num=min_num, 
                    step_size=step_size, 
                    order=order,
                )

            curr_task['id'] = id
            id += 1
            curr_task = {'id': curr_task.pop('id'), **curr_task}
            
            if args.use_placeholders:
                write_data(f'data/numerical_placeholder.jsonl', curr_task)
            else:
                write_data(f'data/numerical.jsonl', curr_task)


if __name__ == "__main__":
    generate_numerical_dataset()