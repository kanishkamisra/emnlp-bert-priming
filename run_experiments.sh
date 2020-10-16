#!/bin/bash

declare -a models=("bert-base-uncased" "bert-large-uncased")
declare -a modes=("sentence" "word")

for model in ${models[@]}
do
    echo "Computing constraint scores and entropies for ${model}"
    python python/constraint.py --model ${model} --infile data/raw_stimuli.csv --outfile data/constraints_${model}.csv

    for mode in ${modes[@]}
    do
        echo "Running priming experiments for ${mode} primes"
        python python/priming.py --model ${model} --mode ${mode} --infile data/raw_stimuli.csv --outfile data/priming_results/priming_${model}_${mode}.csv

        echo "Experiment results saved in data/priming_results/priming_${model}_${mode}.csv!"
    done
done
