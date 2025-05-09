# -*- coding: utf-8 -*-
"""echo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Vb_jCskD1CZrYdZGICvqxUyinEItMw7U
"""

!nvidia-smi

!pip install --upgrade datasets transformers accelerate trl[sft]

from datasets import load_dataset

# 2.1 Load
tldr_ds = load_dataset("trl-lib/tldr")

# 2.2 Inspect splits and column names
print(tldr_ds)
# DatasetDict({
#     train: Dataset({ features: ['prompt','completion'], num_rows: … }),
#     validation: Dataset({ … }),
#     test: Dataset({ … })
# })

print(tldr_ds["train"].column_names)
# ['prompt', 'completion']

# 2.3 Peek at one example
print(tldr_ds["train"][0])

from datasets import load_dataset

tldr_ds   = load_dataset("trl-lib/tldr")
train_ds  = tldr_ds["train"]       # 116,722 examples
valid_ds  = tldr_ds["validation"]  #   6,447 examples
test_ds   = tldr_ds["test"]        #   6,553 examples

print(train_ds.column_names)  # ['prompt', 'completion']
print(train_ds.num_rows, valid_ds.num_rows, test_ds.num_rows)

from transformers import AutoTokenizer

model_name = "google/flan-t5-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(batch):
    # inputs
    tok = tokenizer(
        batch["prompt"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    # targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["completion"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    tok["labels"] = labels["input_ids"]
    return tok

# Use batched=True so this streams and doesn’t error
train_tok = train_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    remove_columns=train_ds.column_names
)
valid_tok = valid_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    remove_columns=valid_ds.column_names
)
test_tok = test_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    remove_columns=test_ds.column_names
)

from datasets import load_dataset

# Load the prefs split you want
prefs = load_dataset("UCL-DARK/openai-tldr-summarisation-preferences")["train"]

# Inspect to confirm…
print(prefs.column_names)  # ['info','split','summaries','choice','worker','batch']
print(prefs[0]['info'])    # to see the fields under info

# Remap using info['title'] & info['post'], and pick the summary at index choice
def preprocess_prefs(example):
    info    = example["info"]
    title   = info["title"]
    post    = info["post"]
    prompt  = f"TITLE: {title}\nPOST: {post}\nTL;DR:"
    # use the worker’s preferred summary
    summary = example["summaries"][example["choice"]]["text"].strip()
    return {"prompt": prompt, "completion": summary}

# Apply
test_prefs = prefs.map(
    preprocess_prefs,
    batched=False,
    remove_columns=prefs.column_names
)

# Verify
print(test_prefs.column_names, test_prefs.num_rows)
print(test_prefs[0])

# Question 6: Tokenize UCL-DARK held-out test set
test_prefs_tok = test_prefs.map(
    tokenize_batch,
    batched=True,
    batch_size=500,
    remove_columns=test_prefs.column_names
)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="tuned_tldr",
    per_device_train_batch_size=8,    # ↑ from 4
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,    # ↓ from 4
    num_train_epochs=3,
    learning_rate=3e-5,
    bf16=True,                        # switched from fp16
    logging_steps=100,
    save_total_limit=2,
    dataloader_num_workers=4,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 4.1 Fine-tune
trainer.train()

# 4.2 Manual evaluation
print("Validation:",   trainer.evaluate(valid_tok))
print("Held-out test:", trainer.evaluate(test_prefs_tok))

trainer.save_model("tuned_tldr/final")
tokenizer.save_pretrained("tuned_tldr/final")

# 1. Create a clean folder (optional if it doesn’t already exist)
import os
os.makedirs("saved_model", exist_ok=True)

# 2. Save the model weights + config
trainer.save_model("saved_model")       # writes pytorch_model.bin, config.json, etc.

# 3. Save the tokenizer files
tokenizer.save_pretrained("saved_model")  # writes vocab, merges (if BPE), tokenizer_config.json, etc.

# 1. Re-save to ensure folder is up-to-date
trainer.save_model("saved_model")
tokenizer.save_pretrained("saved_model")

# 2. Create a ZIP archive of the folder
!zip -r saved_model.zip saved_model

# 3. Download the ZIP to your local machine
from google.colab import files
files.download("saved_model.zip")