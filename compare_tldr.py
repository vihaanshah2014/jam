from transformers import AutoTokenizer, PegasusForConditionalGeneration
from datasets import load_dataset
import requests

# Load Pegasus tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-x-large")

# Lambda API configuration
LAMBDA_URL = "https://h6t24oqevpwinchdw5xwkuta6q0bpocu.lambda-url.us-east-2.on.aws/api/generate"
HEADERS = {"Content-Type": "application/json"}

def generate_pegasus_summary(prompt):
    try:
        inputs = tokenizer(prompt, truncation=True, padding="longest", return_tensors="pt")
        summary_ids = model.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    except Exception as e:
        return f"An error occurred in Pegasus: {e}"

def generate_lambda_response(prompt):
    try:
        payload = {
            "question": prompt,
            "max_length": 50,
            "temperature": 0.1,
            "num_beams": 4
        }
        response = requests.post(LAMBDA_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "").strip()
        return f"API Error: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {e}"

def evaluate_summaries(pegasus_summary, lambda_summary, reference):
    try:
        from openai import OpenAI
        client = OpenAI(api_key="")

        evaluation_prompt = f"""Compare these two summaries against the reference summary and determine which is better.
        Consider accuracy, completeness, and clarity.

        Reference Summary: {reference}

        Summary 1 (Pegasus): {pegasus_summary}
        Summary 2 (Lambda): {lambda_summary}

        just output A or B NOTHING ELSE"""

        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Evaluation error: {e}"

# Test Lambda API first
print("Testing Lambda API with 'hi'...")
test_response = generate_lambda_response("hi")
print(f"Lambda API test response: {test_response}")
print("-" * 80)

# Load dataset - using train split
dataset = load_dataset("davanstrien/dataset-tldr", split="train[:50]")

# Initialize counters
total_a = 0
total_b = 0
total_evaluations = 0

# Run 5 times
for run in range(5):
    print(f"\n=== Run {run + 1} ===")
    run_a = 0
    run_b = 0

    # Process each example
    for idx, example in enumerate(dataset):
        prompt = example["parsed_card"]
        reference = example["tldr"]

        # Generate summaries using Pegasus and Lambda
        pegasus_summary = generate_pegasus_summary(prompt)
        lambda_summary = generate_lambda_response(prompt)

        # Evaluate the summaries
        evaluation = evaluate_summaries(pegasus_summary, lambda_summary, reference)

        # Count A vs B
        if evaluation == "A":
            total_a += 1
            run_a += 1
        elif evaluation == "B":
            total_b += 1
            run_b += 1

        total_evaluations += 1

        # Print input and outputs
        print(f"\nExample {idx + 1}:")
        print("Input:", prompt)
        print("\nPegasus Summary:", pegasus_summary)
        print("\nLambda API Summary:", lambda_summary)
        print("\nReference Summary:", reference)
        print("\nGPT-4 Evaluation:", evaluation)
        
        # Show running totals after each comparison
        print(f"\n--- Running Totals ---")
        print(f"Current run: Pegasus (A): {run_a}, Lambda (B): {run_b}")
        print(f"Overall: Pegasus (A): {total_a}, Lambda (B): {total_b}")
        print(f"Win rate: Pegasus: {total_a/total_evaluations:.2%}, Lambda: {total_b/total_evaluations:.2%}")
        print("-" * 80)
    
    # Print run summary
    print(f"\n=== Run {run + 1} Summary ===")
    print(f"Pegasus (A) wins in this run: {run_a}")
    print(f"Lambda (B) wins in this run: {run_b}")
    print(f"Run {run + 1} win rate: Pegasus: {run_a/len(dataset):.2%}, Lambda: {run_b/len(dataset):.2%}")

# Print final counts
print("\n=== Final Results ===")
print(f"Pegasus (A) wins: {total_a}")
print(f"Lambda (B) wins: {total_b}")
print(f"Total evaluations: {total_evaluations}")
print(f"Final win rate: Pegasus: {total_a/total_evaluations:.2%}, Lambda: {total_b/total_evaluations:.2%}")
