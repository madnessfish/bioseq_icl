import replicate
import os
import re
import pandas as pd
import numpy as np
import argparse
import datetime
from tqdm import tqdm
import json
import time

parser = argparse.ArgumentParser(description="Few-shot prediction",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input dataframe name")
parser.add_argument("-o", "--output_dir", type = str, help="output directory name")
parser.add_argument("-l", "--model_name", type = str, help="Input selected model name")
parser.add_argument("-a", "--api", type = str, help="Input api method", default='chat.completion')

args = parser.parse_args()
input_name = args.input
input_path=f'{args.input_dir}/{input_name}_prompt.json'
output_dir=args.output_dir
model = args.model_name
api_method = args.api
runs = 5
tempt = 0.2

if 'galactica' in model:
    model_date = ''
elif 'llama-2' in model:
    model_date = 'july_2023'
elif 'llama-3-8b' in model:
    model_date = 'march_2023'
elif 'llama-3-70b' in model:
    model_date = 'dec_2023'
elif 'mistral' in model:
    model_date = 'dec_2023'
elif 'mixtral-8x7b' in model:
    model_date = 'dec_2023'

output_name = f'{input_name}_{model}_{model_date}'
output_name = re.sub('/', '_', output_name)

with open(input_path, 'r') as p_file:
    prompt_list = json.load(p_file)

def generate_response_by_replicate(prompt, model_engine):
    if 'llama-2' in model_engine:
        output = replicate.run(
            model_engine,
            input={
                "top_k": 0,
                "top_p": 1,
                "prompt": prompt,
                "temperature": tempt,
                "length_penalty": 1,
                "max_new_tokens": 2048,
                "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
                "presence_penalty": 0,
            }
        )
    elif 'llama-3' in model_engine:
        output = replicate.run(
            model_engine,
            input={
                "top_k": 0,
                "top_p": 1,
                "prompt": prompt,
                "temperature": tempt,
                "length_penalty": 1,
                "max_new_tokens": 4096,
                "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "presence_penalty": 0,
            }
        )
    elif 'mistral-7b' in model_engine:
        output = replicate.run(
            "mistralai/mistral-7b-instruct-v0.2",
            input={
                "top_k": 50,
                "top_p": 0.9,
                "prompt": prompt,
                "temperature": tempt,
                "length_penalty": 1,
                "max_new_tokens": 500,
                "prompt_template": "<s>[INST] {prompt} [/INST] ",
                "presence_penalty": 0
            }
        )
    elif 'mixtral' in model_engine:
        output = replicate.run(
            "mistralai/mixtral-8x7b-instruct-v0.1",
            input={
                "top_k": 50,
                "top_p": 0.9,
                "prompt": prompt,
                "temperature": tempt,
                "length_penalty": 1,
                "max_new_tokens": 500,
                "prompt_template": "<s>[INST] {prompt} [/INST] ",
                "presence_penalty": 0
            }
        )
    output = ''.join(output)
    session_id = ''
    return output, session_id


predict_file = f'{output_dir}/{output_name}_pred'
log_file = f'{output_dir}/{output_name}.log'
print(predict_file)

test_list = [prompt.split('\n')[-3].split(' ')[-1] for prompt in prompt_list]

full_df = pd.DataFrame()
for i in range(1,runs+1):
    if os.path.isfile(f'{predict_file}_{i}.csv') and pd.read_csv(f'{predict_file}_{i}.csv').shape[0]==100:
        result_df = pd.read_csv(f'{predict_file}_{i}.csv')
    else:
        generated_response = []
        for idx, prompt in tqdm(enumerate(prompt_list)):
            current_runs = idx+1
            result_df = pd.DataFrame(test_list[0:idx+1], columns=['input'])
            generated_p, session_id = generate_response_by_replicate(prompt, model)
            with open(log_file, "a") as file:
                now = datetime.datetime.now()
                date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write("=" * 30 + date_time_str + "| Session ID: " + session_id + "=" * 30 + "\n")
                file.write(prompt + "\n")

            print(generated_p)
            generated_response.append(generated_p)
            result_df[f'pred_{i}'] = generated_response
            print(result_df)
            result_df.to_csv(f'{predict_file}_{i}.csv', index=False)

    full_df = pd.concat([full_df, result_df], axis=1)

full_df = full_df.loc[:, ~full_df.columns.duplicated()]
if not full_df.isna().any().any():
    full_df.to_csv(f'{predict_file}_all.csv', index=False)
else:
    print('The dataframe contains NaN, please check')