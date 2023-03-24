import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class StringDataset(Dataset):
    def __init__(self, num_samples, max_len):
        self.num_samples = num_samples
        self.max_len = max_len
        self.samples = []
        self.labels = []
        for i in range(num_samples):
            # 随机生成字符串
            sample = ''.join([str(random.randint(0, 4)) for _ in range(random.randint(1, max_len))])
            self.samples.append(sample)
            # 将字符串转换为One-Hot编码
            label = [0] * 5
            for s in sample:
                label[int(s)] = 1
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class StringTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StringTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (seq_len, batch_size)
        x = self.embedding(x)
        # x: (seq_len, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        # x: (batch_size, seq_len, hidden_size)
        x = self.transformer(x, x)
        # x: (batch_size, seq_len, hidden_size)
        x = x.permute(1, 0, 2)
        # x: (seq_len, batch_size, hidden_size)
        x = self.fc(x)
        # x: (seq_len, batch_size, output_size)
        return x


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=0.001, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for i, (inputs, targets) in enumerate(self.train_loader):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}')

    def evaluate(self, loader):
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, targets).item()
        loss /= len(loader)
        return loss

    def test(self):
        test_loss = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}')


# 创建数据集
train_dataset = StringDataset(num_samples=10000, max_len=10)
val_dataset = StringDataset(num_samples=1000, max_len=10)
test_dataset = StringDataset(num_samples=1000, max_len=10)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建模型
model = StringTransformer(input_size=5, hidden_size=32, output_size=5)

# 创建训练器
trainer = Trainer(model, train_loader, val_loader, test_loader, lr=0.001, epochs=10)

# 训练模型
trainer.train()

# 测试模型
trainer.test()
