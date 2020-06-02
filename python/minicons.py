from transformers import AutoTokenizer, AutoModel
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re

from typing import Optional, Callable

def mask(sentence: str, word: str) -> str:
    replaced = re.sub(rf'(?<![\w\/-])({word})(?=[^\w\/-])', '[MASK]', sentence)
    masked = ['[CLS]'] + [replaced] + ['[SEP]']
    return ' '.join(masked)

def get_batch(train_data, batch_size, shuffle = False):
    if shuffle:
        random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def metrics(scores: torch.Tensor, indices: torch.Tensor):
    probs = F.softmax(scores, dim = 1)
    shape = scores.shape
    inv_ranks = (probs).argsort().argsort() + 1
    # percentiles = inv_ranks.type(torch.FloatTensor)/shape[1]
    ranks = shape[1] - inv_ranks + 1
    word_probs = probs[torch.arange(shape[0]), indices].tolist()
    # word_percentiles = percentiles[torch.arange(shape[0]), indices].tolist()
    word_ranks = ranks[torch.arange(shape[0]), indices].tolist()
    
    return(word_probs, word_ranks)

def highest_prob(scores: torch.Tensor):
    probs = F.softmax(scores, dim = 1)
    highest, idx = torch.max(probs, 1)
    return highest.tolist()

def batch_highest(outputs, current_batch_size, mask):
    distribution = outputs[0][torch.arange(current_batch_size), torch.tensor(mask)]
    decoded = highest_prob(distribution)
    return(decoded)

# Returns the index of the masked token. after applying the model's tokenizer.
def get_mask(context: str, tokenizer: Optional) -> int:
    context_parts = context.split("[MASK]")
    part1 = tokenizer.tokenize(context_parts[0])
    mask_index = len(part1)
    return(mask_index)

def create_inputs(sentence, tokenizer, max_len = 30):
    tokenized_text = tokenizer.tokenize(sentence)
    tokenized_text = tokenized_text + ['[PAD]']*(max_len-len(tokenized_text))
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * max_len
    
    attention_masks = [1 if token != 0 else 0 for token in indexed_tokens]
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    attnmask_tensor = torch.tensor([attention_masks])

    return(tokens_tensor, segments_tensor, attnmask_tensor)

def batch_inputs(tensor_inputs):
    zipped = list(zip(*tensor_inputs))
    return(torch.cat(zipped[0]), torch.cat(zipped[1]), torch.cat(zipped[2]))

def batch_preprocess(inputs, tokenizer, backend_device, max_len = 30):
    inputs = [create_inputs(x, tokenizer, max_len) for x in inputs]
    tokens, segments, attnmasks = batch_inputs(inputs)
    tokens, segments, attnmasks = [x.to(backend_device) for x in [tokens, segments, attnmasks]]
    return (tokens, segments, attnmasks)

def batch_decode(outputs, target_idx, current_batch_size, mask):
    distribution = outputs[0][torch.arange(current_batch_size), torch.tensor(mask)]
    decoded = metrics(distribution, target_idx)
    return(decoded)

def mask_distribution(outputs, current_batch_size, mask):
    distribution = F.softmax(outputs[0][torch.arange(current_batch_size), torch.tensor(mask)], 1)
    return(distribution)

def batch_infer(inference_data, language_model, tokenizer, batch_size, backend_device: Optional, max_len = 30):
    result = []
    for i, batch in tqdm(enumerate(get_batch(inference_data, batch_size))):
        category, constraint, target, related, unrelated, isolated_context, related_context, unrelated_context, isolated_mask, related_mask, unrelated_mask = zip(*batch)

        current_batch_size = len(target)

        isolated_tokens, isolated_segments, isolated_attnmasks = batch_preprocess(isolated_context, tokenizer, backend_device, max_len)
        related_tokens, related_segments, related_attnmasks  = batch_preprocess(related_context, tokenizer, backend_device, max_len)
        unrelated_tokens, unrelated_segments, unrelated_attnmasks  = batch_preprocess(unrelated_context, tokenizer, backend_device, max_len)

        target_idx = torch.tensor(tokenizer.convert_tokens_to_ids(target))
        target_idx = target_idx.to(backend_device)

        language_model.to(backend_device)

        with torch.no_grad():
            isolated_outputs = language_model(input_ids = isolated_tokens, token_type_ids = isolated_segments, attention_mask = isolated_attnmasks)
            related_outputs = language_model(input_ids = related_tokens, token_type_ids = related_segments, attention_mask = related_attnmasks)
            unrelated_outputs = language_model(input_ids = unrelated_tokens, token_type_ids = unrelated_segments, attention_mask = unrelated_attnmasks)

            isolated_triples = batch_decode(isolated_outputs, target_idx, current_batch_size, isolated_mask)
            related_triples = batch_decode(related_outputs, target_idx, current_batch_size, related_mask)
            unrelated_triples = batch_decode(unrelated_outputs, target_idx, current_batch_size, unrelated_mask)

            res = (list(category), list(constraint), list(target), list(related), list(unrelated), list(isolated_context), list(related_context), list(unrelated_context)) + isolated_triples + related_triples + unrelated_triples
            result.extend([list(x) for x in list(zip(*res))])

    return result


