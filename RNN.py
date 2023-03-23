import torch
import torch.nn as nn
import numpy as np

# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Define custom data
input_data = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 0]])
output_data = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]])

# Define hyperparameters
input_size = 3
hidden_size = 4
output_size = 3
learning_rate = 0.1
epochs = 1000

# Initialize RNN model
rnn = RNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# Train RNN model
for epoch in range(epochs):
    loss = 0
    hidden = rnn.init_hidden()

    for input_tensor, output_tensor in zip(input_data, output_data):
        input_tensor = torch.Tensor(input_tensor).view(1, -1)
        output_tensor = torch.LongTensor(np.argmax(output_tensor))

        # Forward pass
        output, hidden = rnn(input_tensor, hidden)
        loss += criterion(output, output_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
# Test RNN model
test_input = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
hidden = rnn.init_hidden()

for input_tensor in test_input:
    input_tensor = torch.Tensor(input_tensor).view(1, -1)
    output, hidden = rnn(input_tensor, hidden)
    print(f"Input: {input_tensor}, Output: {output}")
print("Training complete!")
