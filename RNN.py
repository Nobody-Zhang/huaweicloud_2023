from io import open
import os
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt

# file path: ./RNN_Train_in
# read all files


# -------------load data----------------
all_data = "01234"
n_data = len(all_data)

category_lines = {}
all_categories = []

data_folder = './RNN_Train_in'

# A list of all file paths in the training data folder.
data_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

for FileName in data_paths:
    # reads each file in the training data folder, extracts its category name
    # stores its lines of text in the category_lines dictionary.
    category = os.path.splitext(os.path.basename(FileName))[0]
    all_categories.append(category)
    lines = open(FileName).read().strip().split('\n')
    category_lines[category] = lines
n_categories = len(all_categories)


def letterToIndex(letter):
    """Maps a character to its index in the all_data string"""
    return all_data.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    """Converts a single character to a one-hot encoded tensor"""
    tensor = torch.zeros(1, n_data)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    """converts a line of text to a tensor of one-hot encoded characters, with dimensions (line_length x 1 x n_data)"""
    tensor = torch.zeros(len(line), 1, n_data)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes a new instance of the RNN class with the specified input size, hidden size, and output size"""
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        # takes the concatenation of the input and hidden layers as input, and produces the hidden state as output.
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # takes the concatenation of the input and hidden layers as input, and produces the output state as output.
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # transform the output state into a probability distribution over the output classes.
        self.softmax = nn.LogSoftmax(dim=1)
        # Determines the device to use (CPU or GPU) based on whether a GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_data, hidden_state):
        """Defines the forward pass through the neural network. Returning Output and new Hidden state.
        Including: 1. Combining (Input + 1st Linear Hidden Layer) -> 2. Hiding (1st Combined + 2nd Linear Hidden Layer)
        -> 3. Output (2nd Combined + 3rd Linear Output Layer) + 4. Output Dealing (Output + SoftMax)"""
        input_data = input_data.to(self.device)
        hidden_state = hidden_state.to(self.device)
        combined = torch.cat((input_data, hidden_state), 1)
        combined = combined.to(self.device)
        hidden_state = self.i2h(combined)
        output_state = self.i2o(combined)
        output_state = self.softmax(output_state)
        return output_state, hidden_state  # output state and the new hidden state.

    def initHidden(self):
        """Initializes the hidden state to all zeros"""
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_data, n_hidden, n_categories)
rnn = rnn.to(rnn.device)
"""
# testing single letter
input = letterToTensor('1').to(rnn.device)
hidden = torch.zeros(1, n_hidden).to(rnn.device)

output, next_hidden = rnn(input, hidden)

input = lineToTensor('00001110000200000300000010')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)

print(output)

"""


def categoryFromOutput(output_state):
    """Predict category label and index based on the output_state"""
    top_n, top_i = output_state.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

"""
print(categoryFromOutput(output))
"""

def randomChoice(l):
    """choose a random variable from input l"""
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    """randomly select training entities"""
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


def train(category_tensor, line_tensor, device):
    """Model Training"""
    hidden = rnn.initHidden()
    hidden = hidden.to(device)
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()


n_iters = 100000  # training iterations
print_every = 5000  # gap between each print
plot_every = 1000

"""
n_iters = 1000  # training iterations
print_every = 50  # gap between each print
plot_every = 5
correctness = 0
correctness_rate = []
"""

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    """Report time usage"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

device = torch.device("cuda:0")
cnt = 0
for iteration in range(1, n_iters + 1):
    cnt += 1
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor, device)
    current_loss += loss
    """
    correctness += 1 if categoryFromOutput(output)[0] == category else 0
    correctness_rate.append(correctness / iteration)
    """

    # Print iter number, loss, name and guess
    if iteration % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(
            '%d %d%% (%s) %.4f %s / %s %s' % (iteration, iteration / n_iters * 100,
                                              timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iteration % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()
