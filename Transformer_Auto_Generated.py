import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np


# 定义Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, num_layers=2, num_heads=8, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x


# 定义数据读取器
class TextDataset(data.Dataset):
    def __init__(self, data_dir):
        self.data = []
        for i in range(5):
            filename = data_dir + str(i) + '.in'
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    self.data.append((line, i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = [int(c) for c in x]
        x = torch.tensor(x, dtype=torch.long)
        return x, y


# 定义训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_acc += (predicted == targets).sum().item()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc


# 定义验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_acc += (predicted == targets).sum().item()
    val_loss /= len(val_loader)
    val_acc /= len(val_loader.dataset)
    return val_loss, val_acc


# 定义训练过程
def train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10):
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print('训练中...Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

print('Training Finished.')

vocab_size = 5
hidden_size = 32
num_classes = 5
num_layers = 2
num_heads = 4
dropout = 0.1
batch_size = 32
lr = 0.001
num_epochs = 10


train_dataset = TextDataset('RNN_Generated_Training/')
val_dataset = TextDataset('RNN_Generated_Training/')
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)

model = TransformerClassifier(vocab_size, hidden_size, num_classes, num_layers, num_heads, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)

