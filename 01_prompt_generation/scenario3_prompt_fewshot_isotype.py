import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Few-shot prompt generation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input demonstration list")
parser.add_argument("-t", "--test", type = str, help="input test set")
parser.add_argument("-o", "--output_dir", type = str, help="output directory name")
parser.add_argument("-m", "--mode", type = str, help="selected mode: H (heavy chain only) or HL (heavy-light chain)")
args = parser.parse_args()

input_path = f'{args.input_dir}/{args.input}'
output_dir=args.output_dir
mode = args.mode
test_df = pd.read_csv(f'{args.test}_test_set')
output_name = args.input.replace('_demonstrations.json', '')
selected_chain = mode

with open(input_path, 'r') as p_file:
    bcr_examples = json.load(p_file)

def create_prompt(test_example, fewshot_examples, chain):
    if chain == 'heavy_seq':
        charateristics = 'heavy chain isotype'
        label = 'IGHA (0), IGHD (1), IGHG (2), IGHM (3)'
        prompt = f"You are an AI model specializing in immunology. Your task is to predict the {charateristics} using only the amino acid sequences provided. Recent research has suggested that the {charateristics} of antibody can be predicted with antibody sequence alone, without any structural information.\nPlease strictly follow the format, no other information can be provided. Given the amino acid sequences of a antibody, predict the {charateristics} of antibody based on its sequence, by analyzing whether it is {label}. Consider factors such as physiochemical property, PSSM and amino acid frequency to assess the {charateristics}.\nPlease answer with *only* 0, 1, 2 or 3 for the last example. A few examples are provided in the beginning.\n"
        for example in fewshot_examples:
            prompt += f"Antibody: {example[0]}\nHeavy chain isotype: {example[-1]}\n"
        prompt += f"\nNow, predict the {charateristics} for the following antibody sequence based on the criteria above:\n"
        prompt += f"Antibody: {test_example}\nHeavy chain isotype:\n"
    if chain == 'light_seq':
        charateristics = 'light chain type'
        label = 'IGLC (0), IGKC (1)'
        prompt = f"You are an AI model specializing in immunology. Your task is to predict the {charateristics} using only the amino acid sequences provided. Recent research has suggested that the {charateristics} of antibody can be predicted with antibody sequence alone, without any structural information.\nPlease strictly follow the format, no other information can be provided. Given the amino acid sequences of a antibody, predict the {charateristics} of antibody based on its sequence, by analyzing whether it is {label}. Consider factors such as physiochemical property, PSSM and amino acid frequency to assess the {charateristics}.\nPlease answer with *only* 0, 1 for the last example. A few examples are provided in the beginning.\n"
        for example in fewshot_examples:
            prompt += f"Antibody: {example[0]}\nLight chain type: {example[-1]}\n"
        prompt += f"\nNow, predict the {charateristics} for the following antibody sequence based on the criteria above:\n"
        prompt += f"Antibody: {test_example}\nLight chain type:\n"
    return prompt

no_tag_prompt_file = f'{output_dir}/{output_name}_no_tag_prompt.json'

selected_test_df = test_df.loc[:,[selected_chain,'label']]
test_examples = selected_test_df.to_records(index=False)
prompt_list_notag = []
for test, few_shot in tqdm(zip(test_examples, bcr_examples)):
    prompt_list_notag.extend([create_prompt(test[0], few_shot, mode)])
    with open(no_tag_prompt_file, 'w') as f_prompt:
        json.dump(prompt_list_notag, f_prompt)