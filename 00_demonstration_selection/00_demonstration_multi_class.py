import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from Levenshtein import distance

parser = argparse.ArgumentParser(description="Few-shot demonstrations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input dataframe")
parser.add_argument("-o", "--output_dir", type = str, help="output directory name")
parser.add_argument("-m", "--mode", type = str, help="selected mode: H (heavy chain only) or HL (heavy-light chain)")
parser.add_argument("-n", "--num_class", type = int, help="number of input classes", default=4)

args = parser.parse_args()
input_path=f'{args.input_dir}/{args.input}'
output_dir=args.output_dir
train_df=pd.read_csv(f'{input_path}_train_set')
test_df=pd.read_csv(f'{input_path}_test_set')
mode = args.mode
num_class = int(args.num_class)

sample_methods = ['random', 'similar', 'similar_reverse', 'similar_balanced_random']
# sample_nums = [4,8,16,32,48,100,200]
sample_nums = [4,8,16,32,48]

random_state = 42

if mode=="H":
    selected_chain = "heavy_seq"
elif mode=="HL":
    selected_chain = "heavy_light"
elif mode=="naiveness_mouse":
    selected_chain = "seq"
elif mode=="naiveness_rhesus":
    selected_chain = "seq"
else:
    selected_chain = mode

def leven_distmat(train_set, test_instance, sample_size):
    dist = np.asarray([distance(test_instance, train_str) for train_str in train_set[selected_chain]])
    dist_idx = dist.argsort()[:sample_size][::-1]
    return dist_idx

def random_sample_examples(input_train, sample_size, num_class):
    df_list = [input_train[input_train['label']==i] for i in range(0,num_class)]
    selected_fewshot = [df.sample(n=int(sample_size//num_class)) for df in df_list]
    fewshot_df = pd.concat(selected_fewshot)
    bcr = fewshot_df[selected_chain].tolist()
    class_label = fewshot_df["label"].tolist()
    #convert 1 to "Yes" and 0 to "No"" in class_label
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bcr_examples = list(zip(bcr, class_label))
    return bcr_examples

def similar_sample_examples(input_train_df, test_instance, sample_size, mode=''):
    dist_idx = leven_distmat(input_train_df, test_instance, sample_size)
    if mode == 'reverse':
        rev_idx = np.flip(dist_idx)
        similar_fewshot = input_train_df.iloc[rev_idx,:]
    else:
        similar_fewshot = input_train_df.iloc[dist_idx,:]
    bcr = similar_fewshot[selected_chain].tolist()
    class_label = similar_fewshot['label'].tolist()
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bcr_examples = list(zip(bcr, class_label))
    return bcr_examples

def similar_balanced_sample_examples(input_train_df, test_instance, sample_size, num_class):
    df_list = [input_train_df[input_train_df['label']==i] for i in range(0,num_class)]
    selected_idx = [leven_distmat(df, test_instance, sample_size//num_class) for df in df_list]
    selected_fewshot = [df.iloc[idx] for df, idx in zip(df_list, selected_idx)]
    # if mode == 'pos_neg':
    #     similar_fewshot = pd.concat([pos_fewshot, neg_fewshot])
    # elif mode == 'pos':
    #     similar_fewshot = pos_fewshot
    # elif mode == 'neg_pos':
    #     similar_fewshot = pd.concat([neg_fewshot, pos_fewshot])
    # elif mode == 'random':
    similar_fewshot = pd.concat(selected_fewshot)    # Only random mode is available for multi-class
    similar_fewshot = similar_fewshot.sample(frac=1, random_state=random_state)
    bcr = similar_fewshot[selected_chain].tolist()
    class_label = similar_fewshot['label'].tolist()
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bcr_examples = list(zip(bcr, class_label))
    return bcr_examples

for sample_method in sample_methods:
        for sample_num in sample_nums:
            print(f'Processing: {sample_method} Few-shot: {sample_num}')
            output_name = f'{output_dir}/fewshot_{mode}_{sample_num}_{sample_method}'
            demonstrations_file = f'{output_name}_demonstrations.json'
            prompt_file = f'{output_name}_prompt.json'
            selected_test_df = test_df.loc[:,[selected_chain,'label']]
            test_example = selected_test_df.to_records(index=False)
            demo_list = []
            for test in tqdm(test_example):
                if sample_method == 'random':
                    bcr_examples = random_sample_examples(train_df,sample_num, num_class)
                elif sample_method == 'similar':
                    bcr_examples = similar_sample_examples(train_df, test[0], sample_num)
                elif sample_method == 'similar_reverse':
                    bcr_examples = similar_sample_examples(train_df, test[0], sample_num, mode='reverse')
                elif sample_method == 'similar_balanced_random':
                    bcr_examples = similar_balanced_sample_examples(train_df, test[0], sample_num, num_class)
                else:
                    print('The selected method is unavailable.')
                demo_list.append(bcr_examples)
            with open(demonstrations_file, 'w') as f_demo:
                json.dump(demo_list, f_demo)
