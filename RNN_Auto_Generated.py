import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.token_embedding = nn.Embedding(5, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2) # transpose for transformer
        x = self.transformer_encoder(x)
        x = x.mean(dim=0) # mean pooling
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StringDataset(Dataset):
    def __init__(self, num_samples=10000, max_seq_len=100):
        super(StringDataset, self).__init__()
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        seq_len = random.randint(1, self.max_seq_len)
        seq = torch.randint(low=0, high=5, size=(seq_len,))
        label = seq[0]
        return seq, label


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs, torch.tensor(labels)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3

    train_dataset = StringDataset(num_samples=10000, max_seq_len=100)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = TransformerClassifier(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Test the model
    test_dataset = StringDataset(num_samples=1000, max_seq_len=100)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total:.2f}%")

