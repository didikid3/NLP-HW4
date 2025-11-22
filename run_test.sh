#!/bin/bash

#SBATCH --job-name=prompt_engineering
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=validate-small.out

python score_essays.py \
    --limit 10 \
    --input-file train_dev.csv \
    --output-file results.jsonl \
    --evaluate