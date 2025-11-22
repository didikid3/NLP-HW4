# Assignment 4: Prompt Engineering

With all the emphasis put on students using GenAI to do their homework, why not ask whether we could use those systems to _grade_ their homework? This assignment is a short one that has you focus on designing good prompts and seeing how to use LLM inference engines to scale. The assignment has the following learning goals:

* Learn how to design prompts
* Learn how to use different "small" language models
* See how different prompts generalize across models and which models are better
* Learn how to use vLLM as an example of an inference engine.

## Overview

In this assignment, you will use a **Small** Language Model (SLM) via vLLM to automatically score student essays. Your task is to:

1. **Design an effective prompt** that instructs the SLM to score essays on a holistic scale (1-6)
2. **Implement parsing logic** to extract scores from the model's responses (based on how you designed the prompt)
3. **Evaluate your results** against ground truth scores in the dev set and as a class in a Kaggle competition

The assignment uses the `train_dev.csv` dataset, which contains essays with their ground truth scores, allowing you to evaluate your approach. We won't actually be training anything (inference only!) so you can use whichever part of this data you want.

## Dataset

The `train_dev.csv` file contains the following columns:
- `essay_id`: Unique identifier for each essay
- `assignment`: The essay prompt/assignment description that students were given
- `holistic_essay_score`: The ground truth score (1-6 scale) for evaluation
- `full_text`: The essay text to be scored

The `test.csv` file contains the following columns:
- `essay_id`: Unique identifier for each essay
- `assignment`: The essay prompt/assignment description that students were given
- `full_text`: The essay text to be scored

You'll use a SLM to predict the essay score for each essay in the test file and then upload those predictions to Kaggle as a file with these two columns
- `essay_id`
- `holistic_essay_score`

## Setup

### Prerequisites

1. **Python 3.8+** with pip
2. **CUDA-capable GPU** (vLLM requires GPU support)

### Installation

1. Install required packages:
```bash
pip install vllm transformers torch tqdm numpy pandas
```

2. Ensure you have CUDA installed and properly configured. You can verify this with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### File Structure

Make sure you have the following files in your working directory:
- `score_essays.py` - The main script (you'll edit this)
- `train_dev.csv` - The dataset with essays and ground truth scores
- `test.csv` - the dataset with essays that you need to predict

## Assignment Tasks

### Task 1: Write the Prompt (TODO Section 1)

Your first task is to implement the `create_prompt(prompt_id=0)` function in `score_essays.py`. This function should return a prompt:

1. **Explains the task** - Tell the model it's scoring essays on a holistic scale
2. **Includes context** - Provide the assignment prompt that students were given
3. **Includes the essay** - Provide the full essay text to be scored
4. **Specifies the scale** - Explain the 1-6 scoring scale and what each range means
5. **Requests output format** - Ask for the score in a specific format (e.g., JSON, plain number, etc.)

The function has one argument `prompt_id` which allows you to specify which prompt should be returned. This functionality will let you swap between different prompts on the command line when running your experiments in **Task 3**.


**Tips for effective prompts:**
- Be clear and specific about what you want
- Consider asking for justification to help the model think through the scoring
- Consider writing a rubric for what you think good / bad essays are like
- Specify the exact output format you want (this will help with parsing later)
- You might want to include examples of what good/bad essays look like
- Consider using few-shot examples if your model supports them


### Task 2: Implement Score Parsing (TODO Section 2)

Your second task is to implement the `parse_score()` function. This function should extract the numeric score from the model's response text.

Based on your prompt, the model might return responses in various formats:
- Just a number: `"4"`
- JSON: `{"score": 4, "justification": "..."}`
- Text with embedded score: `"The score is 4 because..."`
- Markdown or other formatted text

**Tips for robust parsing:**
- Try multiple parsing strategies (JSON first, then regex patterns, then fallback)
- Validate that extracted scores are in the valid range (1-6)
- Handle edge cases (missing scores, malformed JSON, etc.)
- Consider logging parsing failures to debug issues


## Running the Script

### Basic Usage

Once you've implemented the two TODO sections, you can run the script:

```bash
python score_essays.py --input-file train_dev.csv --output-file results.jsonl
```

### Command-Line Arguments

The script supports several arguments:

**Input/Output:**
- `--input-file`: Path to input CSV file (default: `train_dev.csv`)
- `--output-file`: Path to output JSONL file (default: `essay_scores.jsonl`)
- `--limit`: Limit number of essays to process (useful for testing)

**Prompt Options**
- `--prompt-id `: The index of the prompt to use (default: 0)

**Model Configuration:**
- `--model-name`: Model to use (default: `Qwen/Qwen3-4B`)
- `--cache-dir`: Directory to cache models (default: HuggingFace cache)
- `--tensor-parallel-size`: Number of GPUs to use (default: auto-detect)

**Processing Settings:**
- `--chunk-size`: Number of essays per batch (default: 100)
  - This is how many essays to process with vLLM at once before writing the output. **You want this higher if possible**
- `--use-chat-template`: Use chat template formatting (default: True)
- `--no-chat-template`: Disable chat template (for base models)
- `--system-message`: Custom system message for chat models

**Sampling Parameters:**
- `--temperature`: Sampling temperature (default: 0.3)
  - Lower = more deterministic, Higher = more random
  - For scoring, lower values (0.1-0.5) are recommended
- `--top-p`: Top-p (nucleus) sampling (default: 0.95)
- `--max-tokens`: Maximum tokens to generate (default: 200)

_You need to choose the sampling parameters specific to each model!! See the Huggingface model cards for what's recommended!_

**Evaluation:**
- `--evaluate`: Evaluate predictions against ground truth scores

### Example Commands

**Test with a small subset:**
```bash
python score_essays.py --limit 10 --output-file test_results.jsonl
```

**Run with evaluation:**
```bash
python score_essays.py --input-file train_dev.csv --output-file results.jsonl --evaluate
```

**Test with custom system message:**
```bash
python score_essays.py \
    --system-message "You are an expert English teacher grading student essays." \
    --evaluate
```

## Output Format

The script outputs a JSONL file (one JSON object per line) with the following structure:

```json
{
  "essay_id": "5.40889E+12",
  "assignment": "Today the majority of humans...",
  "ground_truth_score": "3",
  "model_response": "The essay demonstrates...",
  "predicted_score": 3.0,
  "error": null
}
```

Fields:
- `essay_id`: Original essay identifier
- `assignment`: The assignment prompt
- `ground_truth_score`: The true score (for evaluation)
- `model_response`: Raw response from the LLM
- `predicted_score`: Parsed score (your extraction)
- `error`: Error message if processing failed (null if successful)

For the Kaggle test set prediction, you'll want to process this output to generate your csv file to upload. This will let you decide how to handle missing values.

## Evaluation

When you run with the `--evaluate` flag, the script calculates several metrics:

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and true scores
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more
- **Correlation**: Pearson correlation between predictions and ground truth
- **Exact Match Accuracy**: Percentage of predictions that exactly match ground truth
- **Within One Accuracy**: Percentage of predictions within 1 point of ground truth

**Example output:**
```
Evaluation Results
============================================================
num_valid_predictions: 5000
num_total_essays: 5000
mean_absolute_error: 0.8234
root_mean_squared_error: 1.0234
correlation: 0.7123
exact_match_accuracy: 0.3456
within_one_accuracy: 0.7890
============================================================
```

## Task 3: Evaluating multiple models and prompts.

### Part 3.1: Prompt Writing

Write three prompts that you think will have interesting results. One of these prompts should be the best possible prompt you can come up with. The others can also be potentially good but we'll let you try different things that might not work as well (in theory) just to see how they do work. You should update your `create_prompt` method so that these three prompts can be called from the command line.

### Part 3.2: Predicting

Generate predictions for the first 1000 essays in the `train_dev.csv` using each of the three prompts using three SLM:
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://huggingface.co/Qwen/Qwen3-4B
- https://huggingface.co/ibm-granite/granite-4.0-micro

This will give you 9 different outputs

### Part 3.3: Scoring and Reflection

Using the 9 output files, score each of the combinations. Then generate a bar plot where the x-axis is model, y-axis is RMSE, and hue is the prompt.

Write a short reflection on what you see. Discuss the following:
- How did the models differ (if at all)
- How did your prompts differ in performance?
- Did any of your prompts generalize across all three models?

### Part 3.4

Use your best model+prompt combination to score the test set and upload the preditions to Kaggle.


## Notes:

We strongly recommend submitting your jobs to Great Lakes. You can write your SLURM script to have all the models run one after another by adding lines like:

```
python score_essays.py --input-file train_dev.csv --output-file results.model-X.prompt-0.jsonl --prompt-id 0
python score_essays.py --input-file train_dev.csv --output-file results.model-X.prompt-1.jsonl --prompt-id 1
python score_essays.py --input-file train_dev.csv --output-file results.model-X.prompt-2.jsonl --prompt-id 2

You can use the same slurm account as in Homework 3.

## Submission

When submitting your assignment, include:

1. **Your completed `score_essays.py`** with both TODO sections filled in
2. **A brief report** (1 pages) describing:
   - Your prompt design and rationale
   - Your results with figure
   - Analysis of what worked well and what didn't
   - Any interesting observations or challenges
   - Your kaggle username

