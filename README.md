# emnlp-bert-priming
"Cleaned" repository of our code and analysis of the paper "BERT as a Semantic Priming Subject: Exploring BERT's use of Lexical Cues to Inform Word Probabilities in Context"

This readme will be made more descriptive post EMNLP anonymity period.

## Step 1: Get Data

Link to data will be added once anonymity period ends.

### Data Description

Each dataset in the `data/` repository is a .csv file

**Column Descriptions:**

Common across all datasets:

| Column                | Description                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------------------|
| model                 | BERT model used to generate output probabilities and rank                                                       |
| bin                   | Binned Constraint Bin                                                                                           |
| constraint            | Raw constraint score (Averaged Probability of<br>the most expected completion per BERT-base and <br>BERT-large) |
| scenario              | prime context scenario (sentence vs word)                                                                       |
| target                | Target word from SPP                                                                                            |
| related               | Related prime from SPP                                                                                          |
| unrelated             | Unrelated prime from SPP                                                                                        |
| unprimed_context      | Isolated context, free of related/unrelated primes                                                              |
| related_context       | Context primed with appropriate related prime context                                                           |
| unrelated_context     | Context primed with appropriate unrelated prime context                                                         |
| facilitation          | Surprisal(Target \| Unrelated) - Surprisal(Target \| Related)                                                   |
| unprimed_probability  | P(Target \| Unprimed)                                                                                           |
| unprimed_rank         | Rank of target word in unprimed context                                                                         |
| related_probability   | P(Target \| Related)                                                                                            |
| related_rank          | Rank of target word in related context                                                                          |
| unrelated_probability | P(Target \| Unrelated)                                                                                          |
| unrelated_probability | Rank of target word in unrelated context                                                                        |
## Step 2: Run experiments

```
bash python/run_word_experiments.sh
bash python/run_sentence_experiments.sh
```