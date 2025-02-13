import torch
import torch.nn as nn
import torch.optim as optim
from model import HarryPotterLM
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data_size = len(self.text) - sequence_length
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        sequence = self.text[idx:idx + self.sequence_length]
        target = self.text[idx + 1:idx + self.sequence_length + 1]
        
        sequence = torch.tensor([self.char_to_idx[ch] for ch in sequence])
        target = torch.tensor([self.char_to_idx[ch] for ch in target])
        return sequence, target

def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            output, _ = model(sequences)
            
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

def generate_text(model, seed_text, char_to_idx, idx_to_char, length=200, temperature=0.8):
    model.eval()
    current_text = seed_text
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            sequence = torch.tensor([[char_to_idx[ch] for ch in current_text[-100:]]])
            output, hidden = model(sequence)
            
            probs = (output[0, -1] / temperature).softmax(dim=-1)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            current_text += next_char
    
    return current_text

if __name__ == "__main__":
    # Load your Harry Potter text file
    with open("harry_potter.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Hyperparameters
    sequence_length = 100
    batch_size = 64
    embedding_dim = 128
    hidden_dim = 256
    num_epochs = 20
    learning_rate = 0.001
    
    # Create dataset and dataloader
    dataset = TextDataset(text, sequence_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HarryPotterLM(
        vocab_size=len(dataset.chars),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Train the model
    train_model(model, train_loader, num_epochs, learning_rate, device)
    
    # Generate sample text
    seed_text = "Harry Potter"
    generated_text = generate_text(
        model,
        seed_text,
        dataset.char_to_idx,
        dataset.idx_to_char,
        length=500
    )
    print("\nGenerated Text:")
    print(generated_text) 