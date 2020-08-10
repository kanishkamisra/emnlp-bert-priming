import torch
import csv
import argparse
from tqdm import tqdm

from torch.utils.data import dataloader

import minicons
from transformers import BertTokenizer, BertForMaskedLM

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default = 'bert-base-uncased', type = str)
# parser.add_argument("--device", default = "cpu", type = str)
args = parser.parse_args()

model_type = args.model

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def batch_encode(sentences, device = device):
    encoded = tokenizer.batch_encode_plus(sentences, add_special_tokens = False, pad_to_max_length=True)
    token_ids = torch.tensor(encoded['input_ids'])
    token_ids = token_ids.to(device)
    attention_masks = torch.tensor(encoded['attention_mask'])
    attention_masks = attention_masks.to(device)
    return token_ids, attention_masks

def calculate_entropy(logits, masks):
    size = logits.shape[0]
    probs = logits[torch.arange(size), torch.tensor(masks)].softmax(dim = 1)
    entropy = (-probs * probs.log2()).sum(dim = 1)
    return entropy.tolist()

tokenizer = BertTokenizer.from_pretrained(model_type)
model = BertForMaskedLM.from_pretrained(model_type)
model.to(device)
model.eval()

print("Model Loaded!\n")

contexts = []
with open("../data/unique_contexts.csv", "r") as f:
    next(f)
    reader = csv.reader(f)
    for line in reader:
        context, constraint, category = line
        print(context, constraint)
        contexts.append((context, constraint, minicons.get_mask(context, tokenizer)))

contexts_dl = dataloader.DataLoader(contexts, batch_size = 30)

with open("../data/entropies_{}.csv".format(model_type), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["context", "constraint", "entropy"])
    for batch in tqdm(contexts_dl):
        context, constraint, mask = batch
        input_ids, attentions = batch_encode(context)
        output = model(input_ids, attentions)
        entropy = calculate_entropy(output[0], mask)
        writer.writerows(zip(context, constraint, entropy))

