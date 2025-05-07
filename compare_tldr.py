from openai import OpenAI
from datasets import load_dataset
import requests

# Initialize OpenAI client
client = OpenAI(api_key="")

# Lambda API configuration
LAMBDA_URL = "https://h6t24oqevpwinchdw5xwkuta6q0bpocu.lambda-url.us-east-2.on.aws/api/generate"
HEADERS = {"Content-Type": "application/json"}

def generate_gpt_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": "summarize the following text in a single line: " + prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

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

def evaluate_summaries(openai_summary, lambda_summary, reference):
    try:
        evaluation_prompt = f"""Compare these two summaries against the reference summary and determine which is better.
        Consider accuracy, completeness, and clarity.

        Reference Summary: {reference}

        Summary 1 (OpenAI): {openai_summary}
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

# Run 5 times
for run in range(5):
    print(f"\n=== Run {run + 1} ===")

    # Process each example
    for idx, example in enumerate(dataset):
        prompt = example["parsed_card"]
        reference = example["tldr"]

        # Generate summaries using both methods
        openai_summary = generate_gpt_response(prompt)
        lambda_summary = generate_lambda_response(prompt)

        # Evaluate the summaries
        evaluation = evaluate_summaries(openai_summary, lambda_summary, reference)

        # Count A vs B
        if evaluation == "A":
            total_a += 1
        elif evaluation == "B":
            total_b += 1

        # Print input and outputs
        print(f"\nExample {idx + 1}:")
        print("Input:", prompt)
        print("\nOpenAI Summary:", openai_summary)
        print("\nLambda API Summary:", lambda_summary)
        print("\nReference Summary:", reference)
        print("\nGPT-4o Evaluation:", evaluation)
        print("-" * 80)

# Print final counts
print("\n=== Final Results ===")
print(f"OpenAI (A) wins: {total_a}")
print(f"Lambda (B) wins: {total_b}")
print(f"Total evaluations: {total_a + total_b}")
