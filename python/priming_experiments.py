import torch
import csv
from tqdm import tqdm
import numpy as np

import argparse

import minicons
from transformers import BertTokenizer, BertForMaskedLM

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default = "bert_input", type = str)
    parser.add_argument("--outfile", default = None, type = str)
    parser.add_argument("--model", default = 'bert-base-uncased', type = str)
    # parser.add_argument("--device", default = "cpu", type = str)
    args = parser.parse_args()

    input_file = "../data/{}.csv".format(args.infile)
    results_file = "../data/{}.csv".format(args.outfile)
    # device = args.device
    model_type = args.model

    torch.manual_seed(1234)

    bert_tokenizer = BertTokenizer.from_pretrained(model_type)
    bert_model = BertForMaskedLM.from_pretrained(model_type) # for now
    bert_model.eval()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    bert_model.to(device)
    # device = "cpu"
    print("Using Device: {}".format(device))

    print("Reading SPP.")
    spp = []
    with open(input_file, 'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in tqdm(reader):
            category, constraint, target, related, unrelated, isolated_context, related_context, unrelated_context = row
            isolated_mask, related_mask, unrelated_mask = [minicons.get_mask(x, bert_tokenizer) for x in [isolated_context, related_context, unrelated_context]]
            
            spp.append([category, constraint, target, related, unrelated, isolated_context, related_context, unrelated_context, isolated_mask, related_mask, unrelated_mask])

    print("Running Priming Experiments on {} with model = {}:".format(input_file, model_type))

    with open(results_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "constraint", "target", "related", "unrelated", "isolated_context", "related_context", "unrelated_context", "isolated_probability", "isolated_rank", "related_probability", "related_rank", "unrelated_probability", "unrelated_rank"])

        result = minicons.batch_infer(spp, bert_model, bert_tokenizer, 30, device, 50)
        writer.writerows(result)
    
