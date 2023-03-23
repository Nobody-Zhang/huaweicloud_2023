import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden


    def initHidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            h = h.cuda()
        c = c.cuda()
        return (h, c)

input_size = 4
hidden_size = 32
output_size = 4
num_layers = 2
lr = 0.01
num_epochs = 1000

# 训练数据集（假设仅包含 0 1 2 3 这四个字符）
data = [("013", "1"),
        ("320", "0"),
        ("02", "2"),
        ("33", "3"),
        ("103", "2"),
        ("0120", "0"),
        ("13", "1"),

        ("321", "0"),
        ("2", "2"),
        ("03", "3"),
]

# 将字符转化为数字表示
char_to_index = {"0": 0, "1": 1, "2": 2, "3": 3}

# 创建模型
lstm = LSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

# 进行训练
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(data)):
        inputs = torch.zeros(len(data[i][0]), 1, input_size)
        for j in range(len(data[i][0])):
            inputs[j, 0, char_to_index[data[i][0][j]]] = 1
        targets = torch.LongTensor([char_to_index[data[i][1]]])

        hidden = lstm.initHidden(1)
        outputs, hidden = lstm(inputs, hidden)

        loss = criterion(outputs, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, total_loss))

# 进行预测
test_data = ["0023", "321", "3312", "03", "01320"]
with torch.no_grad():
    for s in test_data:
        inputs = torch.zeros(len(s), 1, input_size)
        for i in range(len(s)):
            inputs[i, 0, char_to_index[s[i]]] = 1
        hidden = lstm.initHidden(1)
        outputs, _ = lstm(inputs, hidden)
        _, predicted = torch.max(outputs.data, 1)
        print(s, "=>", predicted.item())