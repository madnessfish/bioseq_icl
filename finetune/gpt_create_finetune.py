from openai import OpenAI
import argparse
import re

parser = argparse.ArgumentParser(description="Few-shot prediction",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input jsonl name")
parser.add_argument("-m", "--model", type = str, help="finetuned model", default='gpt-3.5-turbo-0125')

args = parser.parse_args()
file_name = args.input
model_name = args.model
input_file = f'{args.input_dir}/{file_name}'
digit_pattern = r'\d+'
k_shot = re.findall(digit_pattern, file_name)[0]

client = OpenAI()
print(file_name)

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

# Upload training file
file_output = client.files.create(
  file=open(input_file, "rb"),
  purpose="fine-tune"
)

# Create finetune model
client.fine_tuning.jobs.create(
  training_file=file_output.id, 
  model=model_name,
  suffix=output_suffix,
)