import torch
import torch.nn as nn

class HarryPotterLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(HarryPotterLM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.rnn(embeds, hidden)
        output = self.fc(output)
        return output, hidden 