import torch
import os
import re
import pandas as pd
import numpy as np
import argparse
import datetime
from tqdm import tqdm
import json

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers import set_seed

set_seed(42)

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
elif 'Llama-2' in model:
    model_date = 'july_2023'
elif 'Llama-3-8B' in model:
    model_date = 'march_2023'
elif 'Llama-3-70B' in model:
    model_date = 'dec_2023'
elif 'mistral' in model:
    model_date = 'dec_2023'

output_name = f'{input_name}_{model}_{model_date}'
output_name = re.sub('/', '_', output_name)

with open(input_path, 'r') as p_file:
    prompt_list = json.load(p_file)

def generate_response_by_mistral(prompt, model_engine, output_num):
    tokenizer = AutoTokenizer.from_pretrained(model_engine)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_engine,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    sequences = pipeline(prompt,
    do_sample=True,
    temperature=tempt,
    num_return_sequences=output_num,
    eos_token_id=tokenizer.eos_token_id,
    return_full_text=False,
    max_length=32768
    )
    message = [seq['generated_text'] for seq in sequences]
    return message

def generate_response_by_galai(prompt, model_engine):
    tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
    model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    message = tokenizer.decode(outputs[0])
    return message

def generate_response_by_meta(prompt, model_engine, output_num):
    tokenizer = AutoTokenizer.from_pretrained(model_engine)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_engine,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    prompt = f'<s>[INST]{prompt}[\INST]'
    sequences = pipeline(prompt,
    do_sample=True,
    temperature=tempt,
    num_return_sequences=output_num,
    eos_token_id=tokenizer.eos_token_id,
    max_length=4096, # max lenght of output, default=4096
    return_full_text=False,
    max_new_tokens=4096
    )
    message = [seq['generated_text'] for seq in sequences]
    return message

def generate_response_by_llama3(prompt, model_engine, output_num):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_engine,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    print(prompt)
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        num_return_sequences=output_num,
        eos_token_id=terminators,
        do_sample=True,
        temperature=tempt,
    )
    message = [seq['generated_text'][len(prompt):] for seq in outputs]
    return message

predict_file = f'{output_dir}/{output_name}_pred'
log_file = f'{output_dir}/{output_name}.log'
print(predict_file)

test_list = [prompt.split('\n')[-3].split(' ')[-1] for prompt in prompt_list]
generated_response = []

if 'galactica' in model:
    full_df = pd.DataFrame()
    for i in range(1,runs+1):
        generated_response = []
        for idx, prompt in tqdm(enumerate(prompt_list)):
            current_runs = idx+1
            result_df = pd.DataFrame(test_list[0:idx+1], columns=['input'])
            with open(log_file, "a") as file:
                now = datetime.datetime.now()
                date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write("=" * 30 + date_time_str + "=" * 30 + "\n")
                file.write(prompt + "\n")
            generated_p = generate_response_by_galai(prompt+'<EOS>', model)
            print(generated_p)
            generated_response.append(generated_p)
            result_df[f'pred_{i}'] = generated_response
            print(result_df)
            if current_runs % 10 == 0:
                result_df.to_csv(f'{predict_file}_{i}.csv', index=False)
    full_df = pd.concat([full_df, result_df], axis=1)
else:
    for idx, prompt in tqdm(enumerate(prompt_list)):
        current_runs = idx+1
        test_df = pd.DataFrame(test_list[0:idx+1], columns=['input'])
        if 'Llama-2' in model:
            prompt='<s>[INST]' + prompt + '[/INST]'
            generated_p = generate_response_by_meta(prompt, model, 5)
        elif 'Llama-3' in model:
            generated_p = generate_response_by_llama3(prompt, model, 5)
        elif 'mistral' in model:
            generated_p = generate_response_by_mistral(prompt, model, 5)
        with open(log_file, "a") as file:
            now = datetime.datetime.now()
            date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write("=" * 30 + date_time_str + "=" * 30 + "\n")
            file.write(prompt + "\n")
        print(generated_p)
        generated_response.append(generated_p)
        if current_runs % 10 == 0:
            result_df = pd.DataFrame(generated_response, columns=['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
            final_df = pd.concat([test_df, result_df], axis=1)
            print(final_df)
            final_df.to_csv(f'{predict_file}.csv', index=False)
    result_df = pd.DataFrame(generated_response, columns=['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])
    final_df = pd.concat([test_df, result_df], axis=1)
    print(final_df)
    final_df.to_csv(f'{predict_file}.csv', index=False)