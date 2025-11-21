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

if mode=="H":
    selected_chain = "heavy_seq"
elif mode=="HL":
    selected_chain = "heavy_light"
elif mode=="naiveness_mouse":
    selected_chain = "seq"
elif mode=="naiveness_rhesus":
    selected_chain = "seq"

if "mouse" in args.input:
    organism = 'Mouse'
elif "rhesus" in args.input:
    organism = 'Rhesus'

with open(input_path, 'r') as p_file:
    bcr_examples = json.load(p_file)


def create_prompt(test_example, fewshot_examples):
    prompt = f"You are an AI model specializing in immunology. Your task is to predict the humanness of antibody using only the amino acid sequences provided. Recent research has suggested that the humanness of antibody can be predicted with antibody sequence alone, without any structural information.\nPlease strictly follow the format, no other information can be provided. Given the amino acid sequences of a antibody, predict the humanness of antibody based on its sequence, by analyzing whether it can Human(1) or {organism}(0). Consider factors such as physiochemical property, PSSM and amino acid frequency to assess the humanness of antibody.\nPlease answer with *only* 1 or 0 only for the last example. A few examples are provided in the beginning.\n"
    for example in fewshot_examples:
        prompt += f"Antibody: {example[0]}\nHumanness: {example[-1]}\n"
    prompt += "Let's solve this problem by splitting it into steps.\nStep 1: Please calculate the frequency of each amino acid in the sequence.\nStep 2: Please consider the physiochemical property of each amino acid in the sequence.\n"    
    prompt += "Now, predict the humanness of antibody for the following antibody sequence based on the criteria above:\n"
    prompt += f"Antibody: {test_example}\nHumanness:\n"
    return prompt

no_tag_prompt_file = f'{output_dir}/cot_{output_name}_no_tag_prompt.json'

selected_test_df = test_df.loc[:,[selected_chain,'label']]
test_examples = selected_test_df.to_records(index=False)
prompt_list_notag = []
for test, few_shot in tqdm(zip(test_examples, bcr_examples)):
    prompt_list_notag.extend([create_prompt(test[0], few_shot)])
    with open(no_tag_prompt_file, 'w') as f_prompt:
        json.dump(prompt_list_notag, f_prompt)