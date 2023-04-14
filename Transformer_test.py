import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Transformer_Auto_Generated import Transformer, train, test, StringDataset

# Set hyperparameters
batch_size = 32
num_epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_dataset = StringDataset("RNN_Train_in")
test_dataset = StringDataset("RNN_Train_in")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = Transformer(num_classes=10, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Train and test the model
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_acc = test(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, test_acc={test_acc:.4f}")
