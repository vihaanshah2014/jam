import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

def check_requirements():
    try:
        import sentencepiece
    except ImportError:
        print("Please install required packages:")
        print("pip install sentencepiece transformers")
        exit(1)
    
    try:
        import numpy
        if numpy.__version__.startswith('2'):
            print("Please run: pip install numpy==1.24.3")
            exit(1)
    except ImportError:
        pass

class WikiQADataset(Dataset):
    def __init__(self, max_length=64, max_samples=1000):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.max_length = max_length
        
        print("Loading dataset...")
        dataset = load_dataset("SohamGhadge/casual-conversation")['train']
        
        all_examples = list(dataset)[:max_samples]
        
        self.examples = []
        for item in all_examples:
            if item.get('question') and item.get('answer'):
                # Clean and format the text
                question = item['question'].strip()
                answer = item['answer'].strip()
                
                # Skip if either is too short
                if len(question) < 3 or len(answer) < 3:
                    continue
                    
                self.examples.append({
                    'question': question,
                    'answer': answer
                })
                if len(self.examples) >= max_samples:
                    break
        
        print(f"Loaded {len(self.examples)} valid samples from casual conversation dataset")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        
        input_text = f"question: {item['question']}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            text=item['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def train_model(model, train_loader, num_epochs, learning_rate, device):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 30)
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Progress: {progress:.1f}% | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

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
         early_stopping=True,
         no_repeat_ngram_size=3,  # Prevent repetition
         top_k=50,  # Limit vocabulary choices
         top_p=0.9  # Nucleus sampling
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    check_requirements()
    
    start_time = time.time()
    
    print("Initializing dataset...")
    dataset = WikiQADataset(max_length=64)
    
    if len(dataset) == 0:
        print("Error: No data was generated. Please check your internet connection and try again.")
        exit(1)
        
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        pin_memory=False
    )
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    print("Initializing model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
    for param in model.encoder.parameters():
        param.requires_grad = False
    for i in range(4):
        for param in model.decoder.block[i].parameters():
            param.requires_grad = False

    print("Starting training...")
    train_model(
        model, 
        train_loader, 
        num_epochs=3,  # Increased epochs
        learning_rate=0.001,
        device=device
    )
    
    end_time = time.time()
    training_duration = end_time - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)
    
    print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    print("Saving model and tokenizer...")
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model and tokenizer saved to 'saved_model' directory")
    
    test_questions = [
        "What color is the sky?",
        "Who is the US president?",
        "Who is Descartes?"
    ]
    
    print("\nTesting the model:")
    for question in test_questions:
        answer = generate_answer(
            model, 
            tokenizer, 
            question, 
            device,
            max_length=150,
            temperature=0.7
        )
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    
    print("\nInteractive Question Answering (type 'exit' to quit):")
    while True:
        user_question = input("Your question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(model, tokenizer, user_question, device, max_length=150, temperature=0.7)
        print(f"A: {answer}\n")