'''
Computes the constraint scores of the cloze-stimuli in terms of the 

(1) probability of the most expected word.
(2) entropy of the output distribution.
'''

import torch
import csv
import argparse
from tqdm import tqdm

from torch.utils.data import dataloader

import minicons
from transformers import AutoTokenizer, AutoModelForMaskedLM

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--infile", type = str)
parser.add_argument("--outfile", type = str)
parser.add_argument("--model", default = 'bert-base-uncased', type = str)
# parser.add_argument("--device", default = "cpu", type = str)
args = parser.parse_args()

model_type = args.model

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModelForMaskedLM.from_pretrained(model_type)
model.to(device)
model.eval()

print("Model Loaded!\n")

contexts = []
with open(args.infile, "r") as f:
    next(f)
    reader = csv.reader(f)
    for line in reader:
        target, related, unrelated, relation, context = line 
        contexts.append((target, related, unrelated, relation, context, minicons.get_mask(context, tokenizer)))

contexts_dl = dataloader.DataLoader(contexts, batch_size = 30)

with open(args.outfile, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["target", "related", "unrelated", "relation", "context", "constraint", "entropy"])
    for batch in tqdm(contexts_dl):
        target, related, unrelated, relation, context, mask = batch
        current_batch_size = len(target)
        input_ids, attentions = minicons.batch_encode(context, tokenizer, device)
        output = model(input_ids, attentions)
        entropy = minicons.calculate_entropy(output[0], mask)
        constraint = minicons.batch_highest(output, current_batch_size, mask)
        writer.writerows(zip(target, related, unrelated, relation, context, constraint, entropy))

