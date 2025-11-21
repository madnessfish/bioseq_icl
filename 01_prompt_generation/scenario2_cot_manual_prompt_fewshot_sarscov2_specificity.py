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
    prompt = f"You are an AI model specializing in immunology. Your task is to predict the binding specificity of antibody to the SARS-CoV-2 spike protein using only the amino acid sequences provided. Recent research has suggested that the specificity to antigens of antibody can be predicted with antibody sequence alone, without any structural information.\nPlease strictly follow the format, no other information can be provided. Given the amino acid sequences of a antibody, predict the receptor binding specificity of antibody based on its sequence, by analyzing whether it can bind(1) or cannot bind(0). Consider factors such as physiochemical property, PSSM and amino acid frequency to assess the humanness of antibody.\nPlease answer with *only* 1 or 0 only for the last example. A few examples are provided in the beginning.\n"
    for example in fewshot_examples:
        prompt += f"Antibody: {example[0]}\nBinding: {example[-1]}\n"
    prompt += "Let's solve this problem by splitting it into steps.\nStep 1: Step 1: Please calculate the frequency of each amino acid in the sequence.\n\n"
    prompt += "To solve the problem, we will first calculate the frequency of each amino acid in the provided antibody sequence. The sequence given is:\n\n"
    prompt += "Antibody: QVQLQESGPGLAKPSETLSLTCTVSGYSISSGHYWGWIRQPPGKGLEWIGSIYHSGSTYYNPSLKSRVTISVDTSKNQFSLKLTSVTAADTAVYYCARAEYSNYYYGMDVWGQGTTVTVSSSYELTQPPSVSVSPGQTARITCSGDALPKQYAYWYQQKPGQAPVLVIYKDSERPSGIPERFSGSSSGTTVTLTISGVQAEDEADYYCQSADSSGTYVVFGGGTKLT\n\n"
    prompt += "Step 1: Calculate the frequency of each amino acid in the sequence.\n\n"
    prompt += "1. Count the occurrences of each amino acid.\n"
    prompt += "2. Divide the count by the total length of the sequence to get the frequency.\n\n"
    prompt += "Let's calculate:\n"
    prompt += "Total length of the sequence = 227 amino acids.\n\n"
    prompt += "- A: 13\n- C: 4\n- D: 8\n- E: 9\n- F: 3\n- G: 23\n- H: 2\n- I: 9\n- K: 9\n- L: 13\n- M: 1\n- N: 3\n- P: 13\n- Q: 14\n- R: 6\n- S: 34\n- T: 22\n- V: 17\n- W: 5\n- Y: 19\n\n"
    prompt += "Now, calculate the frequency of each amino acid:\n\n"
    prompt += "- A: 13 / 227 = 0.0573\n- C: 4 / 227 = 0.0176\n- D: 8 / 227 = 0.0352\n- E: 9 / 227 = 0.0396\n- F: 3 / 227 = 0.0132\n- G: 23 / 227 = 0.1013\n- H: 2 / 227 = 0.0088\n- I: 9 / 227 = 0.0396\n- K: 9 / 227 = 0.0396\n- L: 13 / 227 = 0.0573\n- M: 1 / 227 = 0.044\n- N: 3 / 227 = 0.0132\n- P: 13 / 227 = 0.0573\n- Q: 14 / 227 = 0.0617\n- R: 6 / 227 = 0.0264\n- S: 34 / 227 = 0.1498\n- T: 22 / 227 = 0.0969\n- V: 17 / 227 = 0.0749\n- W: 5 / 227 = 0.0220\n- Y: 19 / 227 = 0.0837\n\n"
    prompt += "These frequencies provide a profile of the amino acid composition of the antibody sequence.\n\n"
    prompt += "Step 2: Please consider the physiochemical property of each amino acid in the sequence.\n"
    prompt += "In Step 2, we consider the physiochemical properties of each amino acid in the antibody sequence. These properties include hydrophobicity, charge, size, and polarity, which can influence the interaction between the antibody and its target antigen. Here's a brief overview of the relevant properties for each amino acid type:\n"
    prompt += "1. **Alanine (A)** - Nonpolar, hydrophobic\n2. **Cysteine (C)** - Polar, uncharged, forms disulfide bonds\n3. **Aspartic acid (D)** - Polar, negatively charged\n4. **Glutamic acid (E)** - Polar, negatively charged\n5. **Phenylalanine (F)** - Nonpolar, hydrophobic, aromatic\n6. **Glycine (G)** - Nonpolar, smallest amino acid, provides flexibility\n7. **Histidine (H)** - Polar, positively charged at physiological pH\n8. **Isoleucine (I)** - Nonpolar, hydrophobic\n9. **Lysine (K)** - Polar, positively charged\n10. **Leucine (L)** - Nonpolar, hydrophobic\n11. **Methionine (M)** - Nonpolar, hydrophobic\n12. **Asparagine (N)** - Polar, uncharged\n13. **Proline (P)** - Nonpolar, introduces kinks in peptide chains\n14. **Glutamine (Q)** - Polar, uncharged\n15. **Arginine (R)** - Polar, positively charged\n16. **Serine (S)** - Polar, uncharged\n17. **Threonine (T)** - Polar, uncharged\n18. **Valine (V)** - Nonpolar, hydrophobic\n19. **Tryptophan (W)** - Nonpolar, hydrophobic, aromatic\n20. **Tyrosine (Y)** - Polar, uncharged, aromatic\n\n"
    prompt += "To analyze the physiochemical properties of the sequence, we can look at the distribution and frequency of these amino acids and infer how they might contribute to the antibody's binding specificity:\n- **Hydrophobicity**: A high frequency of nonpolar, hydrophobic residues (A, F, I, L, M, V, W) can suggest strong interactions with hydrophobic regions of the antigen.\n- **Charge**: The presence of charged residues (D, E, H, K, R) can facilitate interactions with oppositely charged areas on the antigen, influencing specificity and binding strength.\n- **Polarity and Flexibility**: Polar (C, N, Q, S, T, Y) and small or flexible residues (G, P) can affect how the antibody folds and how it interacts with the antigen's surface.\n\nGiven the sequence provided, we can observe a balanced mix of hydrophobic and polar residues, along with a significant presence of charged amino acids. This combination might suggest a versatile binding capability, potentially enhancing the antibody's ability to specifically recognize and bind to diverse regions on the SARS-CoV-2 spike protein.\n"
    prompt += "Now, predict the receptor binding specificity for the following antibody sequence based on the criteria above:\n"
    prompt += f"Antibody: {test_example}\nBinding:\n"
    return prompt

no_tag_prompt_file = f'{output_dir}/cot_manual_{output_name}_no_tag_prompt.json'

selected_test_df = test_df.loc[:,[selected_chain,'label']]
test_examples = selected_test_df.to_records(index=False)
prompt_list_notag = []
for test, few_shot in tqdm(zip(test_examples, bcr_examples)):
    prompt_list_notag.extend([create_prompt(test[0], few_shot)])
    with open(no_tag_prompt_file, 'w') as f_prompt:
        json.dump(prompt_list_notag, f_prompt)