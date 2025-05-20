'''
This file contains all the code to generate datasets of numerical tasks for absence-bench. 

Following the setup of the rest of the repo, datasets will be generated as jsonl files.
jsonl files are a simple format to store json objects in a file, one object per line.
'''

import numpy as np
import random
import json
import os
from num2words import num2words
from word2number import w2n
import roman
from typing import *
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


'''
Numerical tasks can be ordered with a specific step size, follow a mathematical pattern, be in reverse order, or be in random order. They can each use a different number of numbers.

Furthemore, we need not use the decimal system. We can use the binary system, the hexadecimal system, or even Roman numerals.
'''

def is_prime(n: int) -> bool:
    """Helper function to check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def generate_numerical_task(n_numbers: int, sequence_type: str, omission_prob: float, min_num: int=0, step_size: int=1, order: str='ascending', system: str="base10") -> Dict[str, Any]:
    '''
    Generates a numerical task with the given parameters.
    A numerical task is a dictionary with the original sequence, the sequence with some numbers omitted, and the metadata.

    Inputs:
        n_numbers: amount of numbers in the sequence
        sequence_type: the type of sequence to generate.
            Currently supported: "arithmetic", "squares", "cubes", "fibonacci", "primes", "uniform_dist"
        omission_prob: the probability of omitting a number from the sequence
        min_num: the minimum number in the sequence. Relevant for everything but fibonacci.
        step_size: The step size between each number. Relevant for arithmetic, squares, cubes sequences.
        order: The order of the sequence. "ascending", "descending", "random"
        system: The system to use. 
            Currently supported: "base10", "binary", "hexadecimal", "roman", or "words"
    
    Output:
        original_sequence: The original sequence
        user_sequence: The sequence with some numbers omitted
        omitted_mask: the mask of the numbers that were omitted from the sequence
        metadata:
            task_type: will always be "numerical"
            min_num: the minimum number in the sequence
            step_size: the step size between each number
            order: the order of the sequence
            system: the system to use
            n_omitted: the numbers that were omitted from the sequence
    '''
    if sequence_type == 'arithmetic':
        base_sequence = np.arange(min_num, min_num + n_numbers * step_size, step_size)
    elif sequence_type == 'squares':
        base_sequence = np.arange(min_num, min_num + n_numbers * step_size, step_size) ** 2
    elif sequence_type == 'cubes':
        base_sequence = np.arange(min_num, min_num + n_numbers * step_size, step_size) ** 3
    elif sequence_type == 'fibonacci':
        assert min_num == 0, "Fibonacci sequence must start at 0"
        base_sequence = np.zeros(n_numbers)
        base_sequence[0] = min_num
        base_sequence[1] = min_num + 1
        for i in range(2, n_numbers):
            base_sequence[i] = base_sequence[i-1] + base_sequence[i-2]
    elif sequence_type == 'primes':
        # Find first prime >= min_num
        start = min_num
        while not is_prime(start):
            start += 1
            
        # Generate list of n_numbers primes starting from start
        base_sequence = np.zeros(n_numbers)
        base_sequence[0] = start
        count = 1
        curr = start + 1
        
        while count < n_numbers:
            if is_prime(curr):
                base_sequence[count] = curr
                count += 1
            curr += 1
    elif sequence_type == 'uniform_dist':
        base_sequence = np.random.randint(min_num, min_num + n_numbers * step_size, n_numbers)
    
    assert len(base_sequence) == n_numbers, "Base sequence length does not match n_numbers"
    
    # for the metadata, we need the original sequence in numeric form to get min, max, etc.
    base_sequence_numeric = base_sequence.copy()

    if order == 'ascending' or order == 'increasing' or order == 'asc':
        base_sequence = np.sort(base_sequence)
    elif order == 'descending' or order == 'decreasing' or order == 'desc':
        base_sequence = np.sort(base_sequence)[::-1]
    elif order == 'random':
        base_sequence = np.random.permutation(base_sequence)
    
    if system == 'base10' or system == 'decimal':
        base_sequence = base_sequence.astype(int)
    elif system in 'binary':
        base_sequence = np.array([bin(int(num))[2:] for num in base_sequence])
    elif system in 'hexadecimal':
        base_sequence = np.array([hex(int(num)) for num in base_sequence])
    elif system == 'roman':
        base_sequence = np.array([roman.toRoman(int(num)) for num in base_sequence])
    elif system == 'words':
        base_sequence = np.array([num2words(int(num)) for num in base_sequence])
    
    # omitted_mask is a boolean array that is True for the numbers that are not omitted
    # ensure that we have at least one number omitted and not every number omitted
    omitted_mask = np.random.binomial(1, 1-omission_prob, len(base_sequence)).astype(bool)
    while np.sum(omitted_mask) == 0 or np.sum(omitted_mask) == len(base_sequence):
        omitted_mask = np.random.binomial(1, 1-omission_prob, len(base_sequence)).astype(bool)
    n_omitted = len(omitted_mask) - np.sum(omitted_mask)
    # Create user sequence by removing omitted numbers from base sequence
    user_sequence = np.array([x for x, mask in zip(base_sequence, omitted_mask) if mask])

    # data for metadata
    min_num = int(np.min(base_sequence_numeric))
    max_num = int(np.max(base_sequence_numeric))
    step_size = step_size if sequence_type == 'arithmetic' else None
    order = order
    system = system
    n_omitted = int(n_omitted)
    
    result = {}
    result['original_sequence'] = base_sequence.tolist()
    result['user_sequence'] = user_sequence.tolist()
    result['omitted_mask'] = omitted_mask.tolist()
    result['omitted_sequence'] = [x for x, mask in zip(base_sequence.tolist(), omitted_mask) if not mask]
    result['metadata'] = {
        'task_type': 'numerical',
        'n_numbers': n_numbers,
        'sequence_type': sequence_type,
        'omission_prob': omission_prob,
        'n_omitted': n_omitted,
        'min_num': min_num,
        'max_num': max_num,
        'step_size': step_size,
        'order': order,
        'system': system,
    }

    return result

def write_task(jsonl_file: str, task: Dict[str, Any]):
    with open(jsonl_file, 'a') as f:
        f.write(json.dumps(task) + '\n')

def generate_numerical_dataset(analysis:bool=False):
    '''
    Generates an entire dataset of numerical tasks.

    Currently only creates arithmetic sequence datasets. 
    Excludes roman numerals because the package only goes to 5000.

    input:
        - analysis: whether to generate dataset for analysis
    
    by default, the dataset would contain less long-sequence data, use only [ascending] order and [base10] system
    '''
    # generate arithmetic sequence type data
    n_numbers = [50, 100, 500, 1000, 5000]
    omission_probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    step_sizes = [1, 2, 4, 7, 13]
    orders = ['ascending', 'descending', 'random']
    systems = ['base10', 'binary', 'hexadecimal', 'words']
    if analysis:
        id = 0
        for n in n_numbers:
            for omission_prob in omission_probs:
                for step_size in step_sizes:
                    for order in orders:
                        for system in systems:
                            # use randint here as we generate arithmetic sequences
                            min_num = random.randint(0, 10000)
                            curr_task = generate_numerical_task(n, 'arithmetic', omission_prob, min_num=min_num, step_size=step_size, order=order, system=system)
                            curr_task['id'] = id
                            id += 1
                            curr_task = {'id': curr_task.pop('id'), **curr_task}
                            write_task('data/numerical_analysis.jsonl', curr_task)
    else:
    # if not analysis, generate the default dataset
        n_numbers = [50, 100, 500, 750, 1000, 1200]
        number_dist = [240, 360, 240, 240, 120]
        id = 0

        # generate a total of 1200 datapoints
        for i in range(len(n_numbers)-1):
            for _ in range(number_dist[i]):
                n = random.randint(n_numbers[i], n_numbers[i+1])
                omission_prob = random.choice(omission_probs)
                step_size = random.choice(step_sizes)
                order = random.choice(orders)
                # only use base10 since the other two would make the sequence 2x-3x longer
                system = 'base10'
                min_num = random.randint(0, 10000)

                curr_task = generate_numerical_task(n, 'arithmetic', omission_prob, min_num=min_num, step_size=step_size, order=order, system=system)
                curr_task['id'] = id
                id += 1
                curr_task = {'id': curr_task.pop('id'), **curr_task}
                write_task('data/numerical_default.jsonl', curr_task)


if __name__ == "__main__":
    write_data = True
    if write_data:
        generate_numerical_dataset()
        exit()
    else:
        exit()
    numerical_or_dates = 'numerical'
    if numerical_or_dates == 'numerical':
        generate_numerical_task(10, 'arithmetic', 0.1, order='ascending', system='base10')

    if numerical_or_dates == 'dates':
        date = datetime.strptime('January 31, 2021', '%B %d, %Y')
        x = generate_date_interval_task(5, 2, 'days', date, step_size=29, order='descending', date_format='%B %d, %Y')
        print(x['original_sequence'])
        print(x['user_sequence'])
        print(x['omitted_mask'])
        print('--------------------------------')
        print(x['metadata'])