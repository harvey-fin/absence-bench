'''
This file contains all the code for generating needles for the additional analysis in Absence Bench
'''
from typing import *
import pandas as pd

def load_data(file_name: str) -> List[str]:
    """Load the file and return a list of needles"""
    df = pd.read_csv(file_name, sep=";")

    ret_list =  [f"{n} is a character in the Harry Potter novel series" for n in list(df["Name"])]
    return ret_list

def main():
    """The main function"""
    load_data("data/harrypotter/Characters.csv")


if __name__ == "__main__":
    main()