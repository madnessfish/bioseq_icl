import torch
import json
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description="Few-shot demonstrations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--input_dir", type = str, help="input directory name")
parser.add_argument("-i", "--input", type = str, help="input embedding")
parser.add_argument("-p", "--plm", type = str, help="input pLM name")
parser.add_argument("-o", "--output_dir", type = str, help="output directory name")
parser.add_argument("-m", "--mode", type = str, help="selected mode: H (heavy chain only) or HL (heavy-light chain)")

args = parser.parse_args()
input_path = f'{args.input_dir}/{args.input}'
output_dir = args.output_dir
plm = args.plm
mode = args.mode

train_emb = torch.load(f'{input_path}_train_set.mono.{plm}.feature.pt').numpy()
pos_train_emb = torch.load(f'{args.input_dir}/pos_{args.input}_train_set.mono.{plm}.feature.pt').numpy()
neg_train_emb = torch.load(f'{args.input_dir}/neg_{args.input}_train_set.mono.{plm}.feature.pt').numpy()

train_y = np.load(f'{input_path}_train_set_label.npy')
pos_train_y = np.load(f'{args.input_dir}/pos_{args.input}_train_set_label.npy')
neg_train_y = np.load(f'{args.input_dir}/neg_{args.input}_train_set_label.npy')

test_emb = torch.load(f'{input_path}_test_set.mono.{plm}.feature.pt').numpy()

sample_methods = ['random', 'similar', 'similar_reverse', 'similar_balanced_posneg', 'similar_balanced_negpos', 'similar_balanced_random']

sample_nums = [2,4,8,16,32,50,100,200]
random_state = 42
random.seed(random_state)

def random_sample_examples(pos_train_emb, neg_train_emb, pos_train_y, neg_train_y, sample_size):
    selected_sample = sample_size//2
    selected_pos_idx = random.sample(range(0, pos_train_emb.shape[0]), selected_sample)
    selected_neg_idx = random.sample(range(0, neg_train_emb.shape[0]), selected_sample)
    pos = pos_train_emb[selected_pos_idx,:]
    neg = neg_train_emb[selected_neg_idx,:]
    pos_y = pos_train_y[selected_pos_idx]
    neg_y = neg_train_y[selected_neg_idx]
    bcr = np.concatenate([pos, neg]).astype(np.float64).round(3).tolist()
    class_label = np.concatenate([pos_y, neg_y]).tolist()
    #convert 1 to "Yes" and 0 to "No"" in class_label
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bcr_examples = list(zip(bcr, class_label))
    return bcr_examples

def cos_sim(input_train_emb, test_instance, sample_size):
    test_emb = test_instance.reshape(1,-1)
    sim_matrix = cosine_similarity(test_emb, input_train_emb)
    min_sim = sim_matrix.argsort()[:,-sample_size:][0]
    return min_sim

def similar_cos_sim_sample_examples(input_train_emb, test_instance, sample_size, mode=''):
    min_sim = cos_sim(input_train_emb, test_instance, sample_size)
    if mode != 'reverse':
        min_sim = np.flip(min_sim)
    bcr = input_train_emb[min_sim].astype(np.float64).round(3).tolist()
    class_label = train_y[min_sim].tolist()
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bcr_examples = list(zip(bcr, class_label))
    return bcr_examples

def similar_balanced_sample_examples(pos_train_emb, neg_train_emb, test_instance, sample_size, mode):
    selected_sample = sample_size//2
    selected_pos_idx = cos_sim(pos_train_emb, test_instance, selected_sample)
    selected_neg_idx = cos_sim(neg_train_emb, test_instance, selected_sample)
    pos_fewshot = pos_train_emb[selected_pos_idx]
    neg_fewshot = neg_train_emb[selected_neg_idx]
    if mode == 'pos_neg':
        bcr = np.concatenate([pos_fewshot, neg_fewshot]).astype(np.float64).round(3).tolist()
        class_label = np.repeat([1,0], selected_sample).tolist()
    # elif mode == 'pos':
    #     bcr = pos_fewshot.tolist()
    elif mode == 'neg_pos':
        bcr = np.concatenate([neg_fewshot, pos_fewshot]).astype(np.float64).round(3).tolist()
        class_label = np.repeat([0,1], selected_sample).tolist()
    elif mode == 'random':
        bcr = np.concatenate([pos_fewshot, neg_fewshot]).astype(np.float64).round(3).tolist()
        class_label = np.repeat([1,0], selected_sample)
        random_idx = random.sample(range(0, sample_size), sample_size)
        bcr = [bcr[idx] for idx in random_idx]
        class_label = class_label[random_idx].tolist()
    else:
        print(f'Your input mode is invalid')
    # class_label = ["Yes" if i == 1 else "No" for i in class_label]
    bcr_examples = list(zip(bcr, class_label))
    return bcr_examples

for sample_method in sample_methods:
        for sample_num in sample_nums:
            print(f'Processing: {sample_method} Few-shot: {sample_num}')
            output_name = f'{output_dir}/fewshot_{mode}_{sample_num}_{sample_method}'
            demonstrations_file = f'{output_name}_demonstrations.json'
            demo_list = []
            for test_instance in tqdm(test_emb):
                if sample_method == 'random':
                    bcr_examples = random_sample_examples(pos_train_emb, neg_train_emb, pos_train_y, neg_train_y, sample_num)
                elif sample_method == 'similar':
                    bcr_examples = similar_cos_sim_sample_examples(train_emb, test_instance, sample_num)
                elif sample_method == 'similar_reverse':
                    bcr_examples = similar_cos_sim_sample_examples(train_emb, test_instance, sample_num, mode='reverse')
                elif sample_method == 'similar_balanced_posneg':
                    bcr_examples = similar_balanced_sample_examples(pos_train_emb, neg_train_emb, test_instance, sample_num, mode='pos_neg')
                # elif sample_method == 'similar_balanced_pos':
                #     bcr_examples = similar_balanced_sample_examples(pos_train_emb, neg_train_emb, test_instance, sample_num, mode='similar_balanced_pos')
                elif sample_method == 'similar_balanced_negpos':
                    bcr_examples = similar_balanced_sample_examples(pos_train_emb, neg_train_emb, test_instance, sample_num, mode='neg_pos')
                elif sample_method == 'similar_balanced_random':
                    bcr_examples = similar_balanced_sample_examples(pos_train_emb, neg_train_emb, test_instance, sample_num, mode='random')
                else:
                    print('The selected method is unavailable.')
                demo_list.append(bcr_examples)
            with open(demonstrations_file, 'w') as f_demo:
                json.dump(demo_list, f_demo)
