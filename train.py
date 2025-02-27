import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import random
import os
import psutil
import argparse
from tqdm import tqdm
import gc

# Ensure CUDA is disabled
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def check_requirements():
    try:
        import sentencepiece
    except ImportError:
        print("Please install required packages:")
        print("pip install sentencepiece transformers tqdm psutil")
        exit(1)
    
    try:
        import numpy
        if numpy.__version__.startswith('2'):
            print("Please run: pip install numpy==1.24.3")
            exit(1)
    except ImportError:
        pass

class CasualConversationDataset(Dataset):
    def __init__(self, tokenizer, max_length=64, max_samples=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Loading Casual Conversation dataset...")
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

class SQuADDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, max_samples=5000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Loading SQuAD dataset...")
        squad_dataset = load_dataset("rajpurkar/squad")['train']
        
        all_examples = list(squad_dataset)[:max_samples]
        
        self.examples = []
        for item in all_examples:
            if item.get('context') and item.get('question') and item.get('answers') and len(item['answers']['text']) > 0:
                # Clean and format the text
                context = item['context'].strip()
                question = item['question'].strip()
                answer = item['answers']['text'][0].strip()
                
                # Skip if any field is too short
                if len(context) < 10 or len(question) < 3 or len(answer) < 1:
                    continue
                    
                self.examples.append({
                    'context': context,
                    'question': question,
                    'answer': answer
                })
                if len(self.examples) >= max_samples:
                    break
        
        print(f"Loaded {len(self.examples)} valid samples from SQuAD dataset")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        
        # For SQuAD, embed context directly in question during training
        input_text = f"question: Given this context: {item['context']} {item['question']}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            text=item['answer'],
            max_length=self.max_length // 2,  # Answers are typically shorter
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

class UltraChatDataset(Dataset):
    CACHE_DIR = "dataset_cache"
    
    def __init__(self, tokenizer, max_length=128, max_samples=3000, stream_loading=True, cache=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.cache = cache
        
        # Create cache directory if it doesn't exist
        if self.cache and not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
        
        # Check for cached dataset
        cache_file = os.path.join(self.CACHE_DIR, f"ultrachat_{max_samples}.pt")
        if self.cache and os.path.exists(cache_file):
            print(f"Loading cached UltraChat dataset from {cache_file}...")
            self.examples = torch.load(cache_file)
            print(f"Loaded {len(self.examples)} cached samples from UltraChat dataset")
            return
        
        print("Loading UltraChat dataset...")
        
        self.examples = []
        
        # Use streaming option to reduce memory usage during loading
        if stream_loading:
            dataset = load_dataset("stingning/ultrachat", streaming=True)
            counter = 0
            
            with tqdm(total=max_samples, desc="Processing UltraChat samples") as pbar:
                for item in dataset['train']:
                    if 'data' in item and isinstance(item['data'], list) and len(item['data']) >= 2:
                        data = item['data']
                        # Process conversation pairs
                        for i in range(0, len(data)-1, 2):
                            if i+1 < len(data):  # Ensure we have a pair
                                question = data[i].strip()
                                answer = data[i+1].strip()
                                
                                # Skip if either is too short
                                if len(question) < 5 or len(answer) < 5:
                                    continue
                                    
                                self.examples.append({
                                    'question': question,
                                    'answer': answer
                                })
                                counter += 1
                                pbar.update(1)
                                
                                if counter >= max_samples:
                                    break
                        
                    # Check if we've collected enough examples
                    if counter >= max_samples:
                        break
                    
                    # Periodically clear memory
                    if counter % 500 == 0:
                        gc.collect()
        else:
            # Original non-streaming implementation (higher memory usage)
            dataset = load_dataset("stingning/ultrachat")['train']
            all_examples = list(dataset)[:max_samples*3]  # Load more than needed to account for filtering
            
            for item in tqdm(all_examples, desc="Processing UltraChat samples"):
                if 'data' in item and isinstance(item['data'], list) and len(item['data']) >= 2:
                    data = item['data']
                    # Process conversation pairs
                    for i in range(0, len(data)-1, 2):
                        if i+1 < len(data):  # Ensure we have a pair
                            question = data[i].strip()
                            answer = data[i+1].strip()
                            
                            # Skip if either is too short
                            if len(question) < 5 or len(answer) < 5:
                                continue
                                
                            self.examples.append({
                                'question': question,
                                'answer': answer
                            })
                            if len(self.examples) >= max_samples:
                                break
                            
                if len(self.examples) >= max_samples:
                    break
        
        print(f"Loaded {len(self.examples)} valid samples from UltraChat dataset")
        
        # Cache the processed dataset
        if self.cache:
            print(f"Caching processed UltraChat dataset to {cache_file}...")
            torch.save(self.examples, cache_file)
    
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

def train_model(model, train_loader, num_epochs, learning_rate, max_grad_norm=1.0, performance_mode='balanced'):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_batches = len(train_loader)
    
    # Adjust training parameters based on performance mode
    if performance_mode == 'power_saver':
        # Lower performance to save battery and reduce heat
        batch_accumulation_steps = 4
        sleep_interval = 0.01  # Small sleep between batches
    elif performance_mode == 'balanced':
        # Default balanced setting
        batch_accumulation_steps = 2
        sleep_interval = 0.005
    elif performance_mode == 'performance':
        # Maximum performance
        batch_accumulation_steps = 1
        sleep_interval = 0
    else:
        batch_accumulation_steps = 2
        sleep_interval = 0.005
    
    print(f"Training with performance mode: {performance_mode}")
    print(f"Batch accumulation steps: {batch_accumulation_steps}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("=" * 30)
        
        # Create progress bar for better tracking
        progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Skip optimizer.zero_grad() until accumulation steps complete
            if batch_idx % batch_accumulation_steps == 0:
                optimizer.zero_grad()
                
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss / batch_accumulation_steps  # Scale loss for accumulation
            loss.backward()
            
            # Only step and clip gradients after accumulation steps
            if (batch_idx + 1) % batch_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            total_loss += loss.item() * batch_accumulation_steps  # Re-scale for logging
            
            # Update the progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item() * batch_accumulation_steps:.4f}"})
            progress_bar.update(1)
            
            # Optional sleep to reduce thermal pressure
            if sleep_interval > 0:
                time.sleep(sleep_interval)
            
            # Collect garbage every 10 batches to manage memory
            if batch_idx % 10 == 0:
                gc.collect()
                
        progress_bar.close()
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Memory cleanup after each epoch
        gc.collect()

def generate_answer(model, tokenizer, question, max_length=100, temperature=0.7, num_beams=5):
    model.eval()
    
    input_str = f"question: {question}"
    inputs = tokenizer.encode(input_str, return_tensors="pt")
    
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

def monitor_system_resources():
    """Return CPU usage percentage and memory usage in GB."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  # Convert to GB
    return cpu_percent, memory_usage

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a T5 model on various QA datasets")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"],
                        help="Size of the T5 model to use (small, base, large)")
    parser.add_argument("--casual_samples", type=int, default=1000,
                        help="Number of casual conversation samples to use")
    parser.add_argument("--squad_samples", type=int, default=2000,
                        help="Number of SQuAD samples to use")
    parser.add_argument("--ultrachat_samples", type=int, default=1000,
                        help="Number of UltraChat samples to use")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005,
                        help="Learning rate for training")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable dataset caching")
    parser.add_argument("--performance_mode", type=str, default="balanced",
                        choices=["power_saver", "balanced", "performance"],
                        help="Performance profile to use for training")
    parser.add_argument("--no_stream", action="store_true",
                        help="Disable dataset streaming (uses more memory)")
    parser.add_argument("--datasets", type=str, default="all",
                        choices=["all", "casual", "squad", "ultrachat", "casual+squad", "casual+ultrachat", "squad+ultrachat"],
                        help="Which datasets to use for training")
    return parser.parse_args()

if __name__ == "__main__":
    check_requirements()
    args = parse_arguments()
    
    start_time = time.time()
    
    print(f"Performance mode: {args.performance_mode}")
    print(f"Using model size: t5-{args.model_size}")
    print(f"Datasets selected: {args.datasets}")
    
    print("Initializing tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(f"t5-{args.model_size}")
    
    print("Loading datasets...")
    datasets = []
    
    # Load selected datasets based on command line arguments
    if args.datasets in ["all", "casual", "casual+squad", "casual+ultrachat"]:
        casual_dataset = CasualConversationDataset(tokenizer, max_length=64, max_samples=args.casual_samples)
        datasets.append(casual_dataset)
    
    if args.datasets in ["all", "squad", "casual+squad", "squad+ultrachat"]:
        squad_dataset = SQuADDataset(tokenizer, max_length=128, max_samples=args.squad_samples)
        datasets.append(squad_dataset)
    
    if args.datasets in ["all", "ultrachat", "casual+ultrachat", "squad+ultrachat"]:
        print("Loading UltraChat dataset (this may take a while)...")
        print("Press Ctrl+C at any time to skip UltraChat and continue with other datasets.")
        
        try:
            ultrachat_dataset = UltraChatDataset(
                tokenizer, 
                max_length=128, 
                max_samples=args.ultrachat_samples,
                stream_loading=not args.no_stream,
                cache=not args.no_cache
            )
            datasets.append(ultrachat_dataset)
        except KeyboardInterrupt:
            print("\nUltraChat dataset loading interrupted. Continuing with other datasets.")
            gc.collect()  # Clean up any partial loading
    
    if not datasets:
        print("Error: No datasets were loaded. Please check your dataset selection.")
        exit(1)
        
    # Combine all loaded datasets
    combined_dataset = ConcatDataset(datasets)
    print(f"Combined dataset created successfully with {len(combined_dataset)} total samples")
    
    # Create DataLoader with appropriate batch size
    train_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )
    
    # CPU-only device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Report initial system resource usage
    cpu_percent, memory_usage = monitor_system_resources()
    print(f"Initial CPU usage: {cpu_percent}%, Memory usage: {memory_usage:.2f} GB")
    
    print("Initializing model...")
    # Set float32 explicitly
    torch.set_default_tensor_type(torch.FloatTensor)
    model = T5ForConditionalGeneration.from_pretrained(f"t5-{args.model_size}")
    
    # Print model size information
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Approximate size in MB
    print(f"Model size: {model_size_mb:.2f} MB")
    
    print("Starting training...")
    try:
        train_model(
            model, 
            train_loader, 
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            performance_mode=args.performance_mode
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
    
    end_time = time.time()
    training_duration = end_time - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)
    
    print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # Report final system resource usage
    cpu_percent, memory_usage = monitor_system_resources()
    print(f"Final CPU usage: {cpu_percent}%, Memory usage: {memory_usage:.2f} GB")
    
    print("Saving model and tokenizer...")
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("Model and tokenizer saved to 'saved_model' directory")
    
    # Create a comprehensive test set
    print("Preparing comprehensive test set...")
    
    # Predefined test questions for specific capabilities
    predefined_questions = [
        {"question": "What color is the sky?"},
        {"question": "Who is the US president?"},
        {"question": "Who is Descartes?"},
        {"question": "What is the capital of France?", "context": "France is a country in Western Europe with several overseas territories and regions. Paris is the capital and most populous city of France."},
        {"question": "What is machine learning?", "context": "Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy."}
    ]
    
    # Sample from loaded datasets
    casual_samples = []
    squad_samples = []
    ultrachat_samples = []
    
    # Sample from Casual Conversation dataset if loaded
    if 'casual_dataset' in locals() and len(casual_dataset.examples) > 0:
        sample_indices = random.sample(range(len(casual_dataset.examples)), min(5, len(casual_dataset.examples)))
        for idx in sample_indices:
            item = casual_dataset.examples[idx]
            casual_samples.append({
                "question": item["question"],
                "dataset": "Casual Conversation"
            })
    
    # Sample from SQuAD dataset if loaded
    if 'squad_dataset' in locals() and len(squad_dataset.examples) > 0:
        sample_indices = random.sample(range(len(squad_dataset.examples)), min(5, len(squad_dataset.examples)))
        for idx in sample_indices:
            item = squad_dataset.examples[idx]
            squad_samples.append({
                "question": item["question"],
                "context": item["context"],
                "dataset": "SQuAD"
            })
    
    # Sample from UltraChat dataset if loaded
    if 'ultrachat_dataset' in locals() and len(ultrachat_dataset.examples) > 0:
        sample_indices = random.sample(range(len(ultrachat_dataset.examples)), min(5, len(ultrachat_dataset.examples)))
        for idx in sample_indices:
            item = ultrachat_dataset.examples[idx]
            ultrachat_samples.append({
                "question": item["question"],
                "dataset": "UltraChat"
            })
    
    # Additional diverse questions
    diverse_questions = [
        {"question": "How does photosynthesis work?"},
        {"question": "What are the differences between Python and JavaScript?"},
        {"question": "What is the theory of relativity?"},
        {"question": "How do I make chocolate chip cookies?"},
        {"question": "What is the meaning of life?"},
        {"question": "How do neural networks learn?"},
        {"question": "What is climate change?"},
        {"question": "Who wrote the book 'Pride and Prejudice'?"},
        {"question": "How far is the moon from Earth?"},
        {"question": "What are black holes?"},
        {"question": "How do vaccines work?"},
        {"question": "Who invented the internet?"},
        {"question": "What caused World War I?"},
        {"question": "How do airplanes fly?"},
        {"question": "What are the benefits of meditation?"},
        {"question": "Given this information: Blockchain is a type of distributed ledger technology (DLT) that consists of growing list of records, called blocks, that are securely linked together using cryptography. How does blockchain technology work?"},
        {"question": "Based on this context: Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations. What is quantum computing?"},
        {"question": "How does the human digestive system work?"},
        {"question": "What's the difference between renewable and non-renewable energy?"},
        {"question": "How do self-driving cars work?"},
        {"question": "What are the main theories of consciousness?"},
        {"question": "Considering that recommendation algorithms are a subclass of information filtering systems that predict preference a user would give to an item and typically use collaborative filtering, content-based filtering, or hybrid approaches, how do recommendation algorithms work?"},
        {"question": "What is the structure of DNA?"},
        {"question": "How do plants respond to their environment?"},
        {"question": "What is the economic concept of supply and demand?"},
        {"question": "How do different cultures celebrate new year?"}
    ]
    
    # Combine all test questions
    test_questions = predefined_questions + casual_samples + squad_samples + ultrachat_samples + diverse_questions
    
    # Modify questions with context to embed context in the question
    modified_test_questions = []
    for test in test_questions:
        if "context" in test and test.get("question"):
            # Embed context into the question
            context = test["context"]
            question = test["question"]
            embedded_question = f"Given this context: {context} {question}"
            
            # Create new test item without separate context
            new_test = {
                "question": embedded_question,
                "original_question": question
            }
            if "dataset" in test:
                new_test["dataset"] = test["dataset"]
            modified_test_questions.append(new_test)
        else:
            modified_test_questions.append(test)
    
    # Replace with modified questions
    test_questions = modified_test_questions        
    random.shuffle(test_questions)  # Shuffle to mix questions from different sources
    
    print(f"\nTesting the model with {len(test_questions)} questions from various sources:")
    
    # Ask user if they want to run the test set
    run_tests = input("Do you want to run the test set now? (y/n): ").lower() == 'y'
    
    if run_tests:
        test_count = 0
        for test in test_questions:
            test_count += 1
            print(f"\n===== Test {test_count}/{len(test_questions)} =====")
            
            # Print source dataset if available
            if "dataset" in test:
                print(f"Source: {test['dataset']}")
            
            # Print original question if available (for questions with embedded context)
            if "original_question" in test:
                print(f"Original question: {test['original_question']}")
            
            answer = generate_answer(
                model, 
                tokenizer, 
                test["question"],
                max_length=150,
                temperature=0.7
            )
            print(f"Q: {test['question']}")
            print(f"A: {answer}")
    else:
        print("Skipping test set. You can run the interactive mode to test your model.")
    
    print("\nInteractive Question Answering (type 'exit' to quit):")
    print("You can include context in your question by starting with 'Given this context: [your context]'")
    
    while True:
        user_input = input("\nYour question (or 'exit'): ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        answer = generate_answer(
            model, 
            tokenizer, 
            user_input,
            max_length=150, 
            temperature=0.7
        )
        print(f"A: {answer}")