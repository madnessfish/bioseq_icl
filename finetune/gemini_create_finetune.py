import google.generativeai as genai
import json
import argparse
import re
import os
from load_creds import load_creds
import pathlib

parser = argparse.ArgumentParser(description="Few-shot prediction",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input jsonl name")

args = parser.parse_args()
file_name = args.input
input_file = f'{args.input_dir}/{file_name}'
digit_pattern = r'\d+'
k_shot = re.findall(digit_pattern, file_name)[0]

creds = load_creds()
genai.configure(credentials=creds)

if 'similar_finetuned' in file_name:
    output_suffix = 's_'+k_shot    
elif 'similar_reverse_finetuned' in file_name:
    output_suffix = 'sr_'+k_shot
elif 'similar_balanced_random_finetuned' in file_name:
    output_suffix = 'sbr_'+k_shot
elif 'similar_balanced_posneg_finetuned' in file_name:
    output_suffix = 'sbp_'+k_shot
elif 'similar_balanced_negpos_finetuned' in file_name:
    output_suffix = 'sbn_'+k_shot
elif 'random_finetuned' in file_name:
    output_suffix = 'r_'+k_shot
elif 'random_balanced' in file_name:
    output_suffix = 'rb_'+k_shot

name = f'generate-{output_suffix}'
prompt_list = []
with open(input_file, 'r') as p_file:
    for line in p_file:
        prompt_list.append(json.loads(line))

operation = genai.create_tuned_model(
    source_model='models/gemini-1.0-pro-001',
    training_data = prompt_list,
    id = name,
    epoch_count = 100,
    batch_size=4,
    learning_rate=0.001,
)