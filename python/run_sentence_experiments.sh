#!/bin/bash

declare -a models=("bert-base-uncased" "bert-large-uncased")
declare -a datasets=("prior_fix" "polysemy_free")

for file in ${datasets[@]}
do
  for model in ${models[@]}
  do
      python priming_experiments.py --infile sentences_${file} --outfile ${model}_sentences_${file} --model ${model}
  done
done
