import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class StringClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StringClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.Transformer(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class StringDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))

# Example usage
input_size = 10
hidden_size = 20
output_size = 2
data = ['hello', 'world', 'goodbye', 'cruel', 'world']
labels = [0, 1, 0, 1, 1]
char_to_idx = {char: i+1 for i, char in enumerate(set(''.join(data)))}
data = [[char_to_idx[char] for char in string] for string in data]
data = nn.utils.rnn.pad_sequence([torch.tensor(string) for string in data], batch_first=True)
dataset = StringDataset(data, torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = StringClassifier(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, dataloader, criterion, optimizer, num_epochs=10)
# Test the model
test_data = ['hello', 'world', 'goodbye', 'cruel', 'world']
test_labels = [0, 1, 0, 1, 1]
test_data = [[char_to_idx[char] for char in string] for string in test_data]
test_data = nn.utils.rnn.pad_sequence([torch.tensor(string) for string in test_data], batch_first=True)
test_dataset = StringDataset(test_data, torch.tensor(test_labels))
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test data: {} %'.format(100 * correct / total))
# Continue training the model for 5 more epochs
train(model, dataloader, criterion, optimizer, num_epochs=5)
# Test the model again
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test data after additional training: {} %'.format(100 * correct / total))
