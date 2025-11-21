import ast
import os
import json
import pandas as pd
import numpy as np
import argparse
import datetime
from tqdm import tqdm

from openai import OpenAI
import anthropic
import google.generativeai as genai

parser = argparse.ArgumentParser(description="Few-shot prediction",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input dataframe name")
parser.add_argument("-o", "--output_dir", type = str, help="output directory name")
parser.add_argument("-l", "--model_name", type = str, help="Input selected model name")
parser.add_argument("-a", "--api", type = str, help="Input api method", default='chat.completion')
parser.add_argument("-t", "--temperature", type = float, help="Temperature of model", default=0.2)

args = parser.parse_args()
input_name = args.input
input_path=f'{args.input_dir}/{input_name}_prompt.json'
output_dir=args.output_dir
model = args.model_name
api_method = args.api
tempt = args.temperature

if "gpt-3.5-" in model:
    model_date = 'sep_2021'
elif model == "gpt-4o-2024-05-13":
    model_date = 'oct_2023'
elif "gpt-4o-mini-2024-07-18" in model:
    model_date = 'oct_2023'
elif model == "gpt-4-turbo-2024-04-09":
    model_date = 'dec_2023'
elif model == "gpt-4-0125-preview":
    model_date = 'dec_2023'
elif model == 'gpt-4-1106-preview':
    model_date = 'apr_2023'
elif model == "gpt-4-0613":
    model_date = 'sep_2021'
elif model == "gpt-4-32k":
    model_date = 'sep_2021'
elif model == "claude-3-opus-20240229":
    model_date = 'aug_2023'
elif model == "claude-3-sonnet-20240229":
    model_date = 'aug_2023'
elif model == "claude-3-haiku-20240307":
    model_date = 'aug_2023'
elif model == "claude-2.1":
    model_date = 'early_2023'
elif model.startswith("gemini"):
    model_date = 'early_2023'

output_name = f'{input_name}_{model}_{model_date}'

with open(input_path, 'r') as p_file:
    prompt_list = json.load(p_file)

def generate_cot_response_by_gpt(prompt, model_engine):
    api_key = os.environ['OPENAI_API_KEY'] #enter your openai api key her
    client = OpenAI(api_key=api_key)
    prompt_fs = prompt.split('\n\n')[0]
    str_to_remove = 'Please answer with *only* 1 or 0 only for the last example.'
    prompt_fs = prompt_fs.replace(str_to_remove,'')
    prompt_test = prompt.split('\n\n')[-1]
    overall_response = [{"role": "user", "content": prompt_fs}, {"role": "user", "content": "Let's solve this problem by splitting it into steps.\nStep 1: Please calculate the frequency of each amino acid in the sequence."}]
    completion = client.chat.completions.create(
      model=model_engine,
      messages = overall_response,
      temperature=tempt
    )
    session_id = completion.id
    step1_response = completion.choices[0].message.content
    overall_response.append({"role":"assistant","content":step1_response})
    overall_response.append({"role": "user", "content":'Step 2: Please consider the physiochemical property of each amino acid in the sequence.'})
    completion = client.chat.completions.create(
      model=model_engine,
      messages = overall_response,
      temperature=tempt
    )
    step2_response = completion.choices[0].message.content
    overall_response.append({"role":"assistant","content":step2_response})
    overall_response.append({"role": "user", "content":f'Please answer with *only* 1 or 0 only for the following example.\n{prompt_test}'})
    completion = client.chat.completions.create(
      model=model_engine,
      messages = overall_response,
      temperature=tempt,
      n=5
    )
    message = completion.choices
    message = [i.message.content.strip() for i in message]
    return overall_response, message, session_id

def generate_response_by_gpt(prompt, model_engine):
    api_key = os.environ['OPENAI_API_KEY'] #enter your openai api key her
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
      model=model_engine,
      messages = [{"role": "user", "content": prompt}],
      temperature=tempt, n=5
    )
    message = completion.choices
    message = [i.message.content.strip() for i in message]
    session_id = completion.id
    return message, session_id

def generate_response_by_completion_gpt(prompt, model_engine):
    api_key = os.environ['OPENAI_API_KEY'] #enter your openai api key her
    client = OpenAI(api_key=api_key)
    completion = client.completions.create(
      model=model_engine,
      prompt=prompt,
      temperature=tempt, n=5
    )
    message = completion.choices
    message = [i.text.strip() for i in message]
    session_id = completion.id
    return message, session_id

def generate_response_by_claude(prompt, model_engine):
    api_key = os.environ['ANTHROPIC_API_KEY'] #enter your openai api key her
    client = anthropic.Anthropic(api_key=api_key)
    completion = client.messages.create(
        max_tokens=128, model=model_engine, temperature=tempt, n=5,
        messages=[{"role": "user", "content": prompt}]
    )
    # message = completion.content[0].text
    message = completion.choices
    message = [i.message.content.strip() for i in message]
    session_id = completion.id
    return message, session_id

predict_file = f'{output_dir}/{output_name}_pred'
log_file = f'{output_dir}/{output_name}.log'
print(predict_file)

test_list = [prompt.split('\n')[-3].split(' ')[-1] for prompt in prompt_list]

generated_response = []
for idx, prompt in tqdm(enumerate(prompt_list)):
    current_runs = idx+1
    test_df = pd.DataFrame(test_list[0:idx+1], columns=['input'])
    if api_method == 'chat.completion':
        generated_p, session_id = generate_response_by_gpt(prompt, model)
    elif api_method == 'cot.chat.completion':
        overall_response, generated_p, session_id = generate_cot_response_by_gpt(prompt, model)
        cot_prompt=[response['content'] for response in overall_response]
        prompt = '\n'.join(cot_prompt)
    elif api_method == 'completion':
        generated_p, session_id = generate_response_by_completion_gpt(prompt, model)
    elif api_method == 'claude':
        generated_p, session_id = generate_response_by_claude(prompt, model)
    else:
        print('Your selected api method is invalid.')
    with open(log_file, "a") as file:
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write("=" * 30 + date_time_str + "| Session ID: " + session_id + "=" * 30 + "\n")
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