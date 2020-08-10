#!/bin/bash

declare -a models=("bert-base-uncased" "bert-large-uncased")

for model in ${models[@]}
do
    python entropy_constraint.py --model ${model}
done