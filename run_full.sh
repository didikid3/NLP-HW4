#!/bin/bash

#SBATCH --job-name=prompt_engineering_full
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=validate-full.out

# PROMPT 1
python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-0-qwen.jsonl \
    --evaluate \
    --prompt-id 0

python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-0-hf.jsonl \
    --evaluate \
    --model-name "HuggingFaceTB/SmoILM3-3B" \
    --prompt-id 0

python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-0-ibm.jsonl \
    --evaluate \
    --model-name "ibm-granite/granite-4.0-micro" \
    --prompt-id 0

# PROMPT 2
python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-1-qwen.jsonl \
    --evaluate \
    --prompt-id 1

python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-1-hf.jsonl \
    --evaluate \
    --model-name "HuggingFaceTB/SmoILM3-3B" \
    --prompt-id 1

python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-1-ibm.jsonl \
    --evaluate \
    --model-name "ibm-granite/granite-4.0-micro" \
    --prompt-id 1
    
# PROMPT 3
python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-2-qwen.jsonl \
    --evaluate \
    --prompt-id 2

python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-2-hf.jsonl \
    --evaluate \
    --model-name "HuggingFaceTB/SmoILM3-3B" \
    --prompt-id 2

python score_essays.py \
    --input-file train_dev.csv \
    --output-file results-2-ibm.jsonl \
    --evaluate \
    --model-name "ibm-granite/granite-4.0-micro" \
    --prompt-id 2