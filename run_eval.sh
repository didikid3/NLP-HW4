#!/bin/bash

#SBATCH --job-name=prompt_engineering_eval
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=validate-eval.out


python score_essays.py \
    --input-file test_student.csv \
    --output-file eval-results-0-qwen.jsonl \
    --evaluate \
    --prompt-id 0

python score_essays.py \
    --input-file test_student.csv \
    --output-file eval-results-1-qwen.jsonl \
    --evaluate \
    --prompt-id 1