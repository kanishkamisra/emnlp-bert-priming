'''
Performs priming experiments on the raw stimuli 
'''
import torch
import csv
import argparse
from tqdm import tqdm

from torch.utils.data import dataloader

import minicons
from transformers import AutoTokenizer, AutoModelForMaskedLM

parser = argparse.ArgumentParser()
parser.add_argument("--infile", default = "../data/raw_stimuli.csv", type = str)
parser.add_argument("--mode", default = "word", type = str)
parser.add_argument("--outfile", default = None, type = str)
parser.add_argument("--model", default = 'bert-base-uncased', type = str)
parser.add_argument("--bs", default = 30, type = int)
# parser.add_argument("--device", default = "cpu", type = str)
args = parser.parse_args()

input_file = args.infile
results_file = args.outfile
mode = args.mode
model_type = args.model

torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained(model_type)
model = AutoModelForMaskedLM.from_pretrained(model_type) # for now
model.eval()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

def prime(sentence, prime, mode):
    if mode == "sentence":
        prefix = f"[CLS] the next word is {prime}."
    else:
        prefix = f"[CLS] {prime}."
    return sentence.replace("[CLS]", prefix)

def batch_metrics(batch_context, word, model, tokenizer, mask, batch_size, device = device, best = True):

    # encode
    ids, attn_masks = minicons.batch_encode(batch_context, tokenizer, device)

    # get logits
    output = model(input_ids = ids, attention_mask = attn_masks)
    logits = output[0]

    if device == 'cuda:0' or device == "cuda:1":
        logits.detach()

    idx = torch.tensor(tokenizer.convert_tokens_to_ids(word))
    idx = idx.to(device)

    distribution = torch.softmax(logits[torch.arange(batch_size), torch.tensor(mask)], 1)
    word_prob, word_rank = minicons.metrics(distribution, idx)

    if best:
        top_k = torch.topk(distribution, 1)
        most_expected = tokenizer.convert_ids_to_tokens(top_k[1])
        top1_prob = top_k[0].flatten().tolist()
        return (word_prob, word_rank, most_expected, top1_prob)

    else:
        return (word_prob, word_rank)

# results_file = "data/priming_results/priming_{}_{}.csv".format(model_type, mode)

# read input file and construct primed instances.
stimuli = []
with open(input_file, "r") as f:
    next(f)
    reader = csv.reader(f)
    for line in reader:
        target, related, unrelated, relation, context = line
        related_context = prime(context, related, mode)
        unrelated_context = prime(context, unrelated, mode)

        isolated_mask, related_mask, unrelated_mask = [minicons.get_mask(x, tokenizer) for x in [context, related_context, unrelated_context]]

        stimuli.append([target, related, unrelated, relation, context, related_context, unrelated_context, isolated_mask, related_mask, unrelated_mask])

stimuli_dl = dataloader.DataLoader(stimuli, num_workers=4, batch_size=args.bs)

'''
From output logits and extract:

1. P(t | c), P(t | r, c), P(t | u, c)
2. argmax(P(x)) in all cases - torch.topk(P(x))
3. max(P(x)) in all cases <- this is same as constraint for isolated context! 

'''
results = []
for batch in tqdm(stimuli_dl):
    target, related, unrelated, relation, context, related_context, unrelated_context, isolated_mask, related_mask, unrelated_mask = batch
    
    current_batch_size = len(target)

    isolated_metrics, related_metrics, unrelated_metrics = [batch_metrics(x, target, model, tokenizer, mask, current_batch_size) for x, mask in [(context, isolated_mask), (related_context, related_mask), (unrelated_context, unrelated_mask)]]

    # Expectations for related and unrelated words in 

    rp_metrics, up_metrics = [batch_metrics(context, x, model, tokenizer, isolated_mask, current_batch_size, best = False) for x in [related, unrelated]]

    result = (target, related, unrelated, relation, context, related_context, unrelated_context) + isolated_metrics + related_metrics + unrelated_metrics + rp_metrics + up_metrics

    results.extend([x for x in list(zip(*result))])

with open(results_file, "w") as f:
    writer = csv.writer(f)
    column_names = ["target", "related", "unrelated", "relation", "context", "related_context", "unrelated_context"]

    column_names.extend([f"{x}_{y}" for x in ["isolated", "related", "unrelated"] for y in ["probability", "rank", "argmax", "maxprob"]])

    column_names.extend([f"{x}_{y}" for x in ["rp", "up"] for y in ['probability', 'rank']])

    writer.writerow(column_names)

    writer.writerows(results)