#!/usr/bin/env python3
"""
benchmark_tldr.py

A script to evaluate your local TL;DR API using the davanstrien/dataset-tldr dataset.
Computes ROUGE and BLEU scores for model-generated summaries.

Dependencies:
    pip install datasets evaluate requests rouge_score sacrebleu
Usage:
    python benchmark_tldr.py
"""

import requests
from datasets import load_dataset
import evaluate

# ---------- Configuration ----------
API_URL = "https://h6t24oqevpwinchdw5xwkuta6q0bpocu.lambda-url.us-east-2.on.aws/api/generate"
HEADERS = {"Content-Type": "application/json"}

# Load a sample (first 50 examples) from the train split
dataset = load_dataset("davanstrien/dataset-tldr", split="train[:50]")

# Inspect to confirm column names
print("Dataset columns:", dataset.column_names)
# Set these to the actual column names
INPUT_COL = "parsed_card"
REF_COL = "tldr"

# Initialize metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

predictions = []
references = []

# Iterate over examples, call the API, collect outputs
for idx, example in enumerate(dataset):
    prompt = example[INPUT_COL]
    reference = example[REF_COL]

    payload = {
        "question": prompt,
        "max_length": 100,
        "temperature": 0.1,
        "num_beams": 4
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    if resp.status_code != 200:
        print(f"[{idx}] ERROR {resp.status_code}: {resp.text}")
        answer = ""
    else:
        answer = resp.json().get("answer", "").strip()

    print(f"[{idx}] Prompt: {prompt[:60]}...")
    print(f"[{idx}] Pred  : {answer[:60]}...")
    print(f"[{idx}] Ref   : {reference[:60]}...\n")

    predictions.append(answer)
    references.append(reference)

# Compute ROUGE scores
rouge_scores = rouge.compute(predictions=predictions, references=references)
print("=== ROUGE Scores ===")
print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}\n")

# Compute BLEU score (requires tokenized lists)
tokenized_preds = [pred.split() for pred in predictions]
tokenized_refs = [[ref.split()] for ref in references]
bleu_scores = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
print("=== BLEU Score ===")
print(f"BLEU:    {bleu_scores['bleu']:.4f}")
