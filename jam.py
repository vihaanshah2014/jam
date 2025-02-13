import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_answer(model, tokenizer, question, device, max_length=50, temperature=0.7, num_beams=5):
    model.eval()
    input_str = f"question: {question}"
    inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)
    
    outputs = model.generate(
         inputs,
         max_length=max_length,
         num_beams=num_beams,
         do_sample=True,
         temperature=temperature,
         early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    device = "cpu"
    print("Loading model and tokenizer...")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained("saved_model")
        model = T5ForConditionalGeneration.from_pretrained("saved_model").to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Make sure you have trained and saved the model first.")
        exit(1)

    print("\nInteractive Question Answering (type 'exit' to quit):")
    while True:
        user_question = input("Your question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(
            model, 
            tokenizer, 
            user_question, 
            device, 
            max_length=150, 
            temperature=0.7
        )
        print(f"A: {answer}\n") 