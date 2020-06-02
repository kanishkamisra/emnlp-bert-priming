from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import codecs
import csv
import numpy as np
from tqdm import tqdm
import random
from itertools import chain
from transformers import BertTokenizer

random.seed(1234)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

story_vocab = defaultdict(int)
spp_story_sentences = defaultdict(list)
stories = []

print("Parsing Sentences from ROC Stories...")
with codecs.open('../data/stories.txt', 'r', 'utf-8') as f:
    for line in tqdm(f):
        tokenized = word_tokenize(line)
        stories.append(tokenized)
        for word in tokenized:
            story_vocab[word] += 1


print("Loading SPP words and storing sentences")
with codecs.open('../data/spp_story_instances.csv', 'r', 'utf-8') as f:
    reader = csv.reader(f)
    next(f)
    for entry in tqdm(reader):
        _, target, _, _, _, _, _, _, _ = entry
        for sentence in stories:
            counts = sum([1 for w in sentence if w == target])
            if counts == 1 and target != "am":
                spp_story_sentences[target].append(sentence)

spp_story_lengths = defaultdict(int)
for k, v in spp_story_sentences.items():
    spp_story_lengths[k] = len(v)

print(f'Average Number of Sentences per SPP word: {np.mean(list(spp_story_lengths.values()))}')
print(f'Minimum Number of Sentences per SPP word: {min(spp_story_lengths.values())}')

with codecs.open("../data/spp_story_lengths.csv", 'w', 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["word", "n"])
    for k, v in spp_story_lengths.items():
        writer.writerow([k, v])

print("Random Sampling!")
with codecs.open("../data/random_sample_spp_stories.csv", 'w', 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["target", "sentence"])
    for k, v in spp_story_sentences.items():
        rs = random.sample(v, min(10, len(v)))
        instances = chain.from_iterable([list(zip([k] * len(rs), rs))])
        for t, s in instances:
            writer.writerow([t, " ".join(s)])

spp = []
with open("../data/spp_story_instances.csv", 'r') as f:
    with open("../data/spp_bert_story.csv", 'w') as fp:
        reader = csv.reader(f)
        writer = csv.writer(fp)
        next(f)
        writer.writerow(["association","target","related","unrelated","relation1","relation2","delta_200ms","delta_1200ms","pos"])
        for line in reader:
            association,target,prime,unrelated,relation1,relation2,delta_200ms,delta_1200ms, pos = line
            t = tokenizer.tokenize(target)
            if len(t) == 1:
                writer.writerow([association,target,prime,unrelated,relation1,relation2,delta_200ms,delta_1200ms, pos])