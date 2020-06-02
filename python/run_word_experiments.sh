#!/bin/bash

declare -a models=("bert-base-uncased" "bert-large-uncased")
declare -a datasets=("prior_fix_stimuli" "polysemy_free")

for file in ${datasets[@]}
do
  for model in ${models[@]}
  do
      python priming_experiments.py --infile ${file} --outfile ${model}_${file} --model ${model}
  done
done
