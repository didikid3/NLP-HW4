#!/bin/bash

#SBATCH --job-name=prompt_engineering
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=validate-small.out

python score_essays.py \
    --limit 1000 \
    --input-file train_dev.csv \
    --output-file results-0.jsonl \
    --evaluate \
    --prompt-id 0

python score_essays.py \
    --limit 1000 \
    --input-file train_dev.csv \
    --output-file results-1.jsonl \
    --evaluate \
    --prompt-id 1

python score_essays.py \
    --limit 1000 \
    --input-file train_dev.csv \
    --output-file results-2.jsonl \
    --evaluate \
    --prompt-id 2