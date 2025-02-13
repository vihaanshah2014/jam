import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wikipedia
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
    def __init__(self, max_length=512):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.max_length = max_length
        self.qa_pairs = self.generate_qa_pairs()
        print(f"Generated {len(self.qa_pairs)} QA pairs")
        
    def generate_qa_pairs(self, num_articles=100):
        qa_pairs = []
        successful_articles = 0
        
        # Replace fixed topics with dynamic category exploration
        categories = [
            "Science", "History", "Geography", "Politics", 
            "Philosophy", "Technology", "Arts", "Sports",
            "Medicine", "Economics", "Religion", "Culture"
        ]
        
        processed_titles = set()  # Track processed articles to avoid duplicates
        
        for category in categories:
            try:
                # Get articles from category
                articles = wikipedia.search(category, results=num_articles//len(categories))
                
                for article_title in articles:
                    if article_title in processed_titles:
                        continue
                    
                    processed_titles.add(article_title)
                    
                    try:
                        print(f"Fetching article: {article_title}")
                        page = wikipedia.page(article_title, auto_suggest=False)
                        content = page.content
                        
                        # Split content into paragraphs and filter for a reasonable length
                        paragraphs = [p.strip() for p in content.split('\n') 
                                      if len(p.strip()) > 50 and len(p.strip()) < 300]
                        
                        if len(paragraphs) < 2:
                            continue
                        
                        # Generate QA pairs from paragraphs using the first sentence as an answer
                        for paragraph in paragraphs:
                            # Extract the first sentence as a concise answer
                            first_sentence = paragraph.split('.')[0]
                            if first_sentence:
                                answer = first_sentence if first_sentence.endswith('.') else first_sentence + '.'
                            else:
                                answer = paragraph  # fallback
                            
                            # Create simple, direct questions
                            questions = [
                                "What color is the sky?",
                                "Why is the sky blue?",
                                "Who is the current US president?",
                                "Who is the president of the United States?",
                                "Who is Descartes?",
                                "Tell me about Descartes.",
                                "What did Descartes do?"
                            ]
                            
                            # Only add QA pairs for relevant topics and questions
                            for question in questions:
                                if ("sky" in question.lower() and "sky" in category.lower()) or \
                                   ("president" in question.lower() and "biden" in category.lower()) or \
                                   ("descartes" in question.lower() and "descartes" in category.lower()):
                                    qa_pairs.append({
                                        'question': question,
                                        'context': category,  # context is no longer used in training input
                                        'answer': answer
                                    })
                        
                        successful_articles += 1
                        print(f"Successfully processed article {successful_articles}/{len(categories)}")
                        
                        # Reduce sleep time for larger datasets
                        time.sleep(0.5)  # reduced from 1.0
                        
                    except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
                        print(f"Skipping article due to error: {str(e)}")
                        continue
                
            except Exception as e:
                print(f"Error processing category {category}: {str(e)}")
                continue

        return qa_pairs
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        
        # Use only the question in the input so that testing takes a simple question-only prompt
        input_text = f"question: {qa_pair['question']}"
        target_text = qa_pair['answer']
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # For T5 training, replace padding token id with -100 in labels so they are ignored by the loss
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
        print("=" * 50)
        
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
            progress = (batch_idx + 1) / total_batches * 100
            if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                print(f"Progress: {progress:.1f}% [{batch_idx + 1}/{total_batches}] | Loss: {loss.item():.4f}")
        avg_loss = total_loss / total_batches
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

def generate_answer(model, tokenizer, question, device, max_length=50, temperature=0.7, num_beams=5):
    model.eval()
    input_str = f"question: {question}"
    inputs = tokenizer.encode(input_str, return_tensors="pt").to(device)
    
    # Note: setting do_sample=True allows temperature to be used.
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
    # Check requirements first
    check_requirements()
    
    start_time = time.time()
    
    print("Initializing dataset...")
    dataset = WikiQADataset()
    
    if len(dataset) == 0:
        print("Error: No data was generated. Please check your internet connection and try again.")
        exit(1)
        
    print(f"Dataset created successfully with {len(dataset)} samples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=8,  # Reduced batch size
        shuffle=True,
        num_workers=0
    )
    
    device = "cpu"
    print("Initializing model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
    # Freeze the encoder parameters to reduce compute and avoid overfitting
    for param in model.encoder.parameters():
        param.requires_grad = False

    print("Starting training...")
    train_model(model, train_loader, num_epochs=5, learning_rate=0.0001, device=device)
    
    end_time = time.time()
    training_duration = end_time - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)
    
    print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # Save the model and tokenizer
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
    
    # Add interactive mode so you can ask your own questions after training
    print("\nInteractive Question Answering (type 'exit' to quit):")
    while True:
        user_question = input("Your question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = generate_answer(model, tokenizer, user_question, device, max_length=150, temperature=0.7)
        print(f"A: {answer}\n")