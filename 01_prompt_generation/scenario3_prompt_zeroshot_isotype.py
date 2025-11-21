import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Few-shot prompt generation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--test", type = str, help="input test set")
parser.add_argument("-o", "--output_dir", type = str, help="output directory name")
parser.add_argument("-n", "--output_name", type = str, help="output name")
parser.add_argument("-m", "--mode", type = str, help="selected mode: H (heavy chain only) or HL (heavy-light chain)")
args = parser.parse_args()

output_dir=args.output_dir
mode = args.mode
test_df = pd.read_csv(f'{args.test}_test_set')
output_name = args.output_name

selected_chain = mode

def create_prompt(test_example):
    if selected_chain == 'heavy_seq':
        charateristics = 'heavy chain isotype'
        label = 'IGHA (0), IGHD (1), IGHG (2), IGHM (3)'
        prompt = f"You are an AI model specializing in immunology. Your task is to predict the {charateristics} using only the amino acid sequences provided. Recent research has suggested that the {charateristics} of antibody can be predicted with antibody sequence alone, without any structural information.\nPlease strictly follow the format, no other information can be provided. Given the amino acid sequences of a antibody, predict the {charateristics} of antibody based on its sequence, by analyzing whether it is {label}. Consider factors such as physiochemical property, PSSM and amino acid frequency to assess the {charateristics}.\nPlease answer with *only* 0, 1, 2 or 3 for the last example.\n"
        prompt += f"Antibody: {test_example}\nHeavy chain isotype:\n"
    return prompt

prompt_file = f'{output_dir}/{output_name}_zero_shot_prompt.json'

selected_test_df = test_df.loc[:,[selected_chain,'label']]
test_examples = selected_test_df.to_records(index=False)
prompt_list = []
for test in tqdm(test_examples):
    prompt_list.extend([create_prompt(test[0])])
    with open(prompt_file, 'w') as f_prompt:
        json.dump(prompt_list, f_prompt)
