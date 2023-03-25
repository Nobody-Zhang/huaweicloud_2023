from io import open
import os
import torch
import random
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr = 0.005):
        super(RNN, self).__init__()

        self.n_categories = None

        self.category_lines = {}
        self.all_categories = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(self.device)

    def loaddata(self, data_folder = './RNN_Train_in', all_data = "01234"):

        n_data = len(all_data)

        data_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

        for FileName in data_paths:
            category = os.path.splitext(os.path.basename(FileName))[0]
            self.all_categories.append(category)
            lines = open(FileName).read().strip().split('\n')
            self.category_lines[category] = lines
        self.n_categories = len(self.all_categories)

    @overload
    def train(self, category_tensor, line_tensor, device):
        criterion = nn.NLLLoss()
        hidden = self.initHidden()
        category_tensor = category_tensor.to(device)
        line_tensor = line_tensor.to(device)
        self.zero_grad()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        loss = criterion(output, category_tensor)
        loss.backward()
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-self.lr)
        return output, loss.item()

    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor