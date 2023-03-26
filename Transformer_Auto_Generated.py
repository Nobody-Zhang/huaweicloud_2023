import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os

# 定义数据集类，继承自Pytorch的Dataset类
class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = int(filename.split('.')[0])
        with open(os.path.join(self.root_dir, filename), 'r') as f:
            data = f.readline().strip()
        return data, label

def collate_fn(batch):
    """
    合并不同大小的 tensor
    """
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    data = [torch.tensor([int(x) for x in item]) for item in data] # 将字符串转换成 tensor
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True) # 对齐长度，使得每个 batch 中所有的 tensor 大小相同
    return data, torch.tensor(label)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model  # 添加这一行代码
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                                       dim_feedforward=dim_feedforward,
                                                                       dropout=dropout),
                                             num_layers=num_encoder_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, 5)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.encoder(src)
        src = src.mean(dim=1)
        out = self.fc(src)
        return out

# 定义位置编码类
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

# 加载数据
batch_size = 16 # 设置 batch_size
dataset = MyDataset('RNN_Train_in')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# 初始化参数
# data_dir = 'RNN_Train_in/'
batch_size = 32
learning_rate = 0.001
num_epochs = 10
d_model = 256
nhead = 8
num_layers = 6
num_classes = 5

# 创建数据集和数据加载器
# dataset = CustomDataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型和优化器
model = TransformerModel(d_model, nhead, num_layers, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 判断是否有GPU资源
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
model.to(device)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        # 将数据和标签移动到GPU上（如果有的话）
        data, target = data.to(device), target.to(device)
        # 将数据转化为整数类型
        data = data.type(torch.LongTensor).to(device)
        # 将梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失函数
        loss = nn.CrossEntropyLoss()(output, target.argmax(dim=1))
        # 反向传播
        loss.backward()
        optimizer.step()
        # 记录总的损失
        total_loss += loss.item()
        # 每100个batch输出一次日志
        if (batch_idx+1) % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(dataset)//batch_size+1, total_loss/(batch_idx+1)))

# 测试模型
test_data = ['01234', '43210', '021', '3210', '1', '']
with torch.no_grad():
    for data in test_data:
        # 将数据转化为数字编码
        data = torch.LongTensor([int(c) for c in data]).unsqueeze(0).to(device)
        # 使用模型进行预测
        output = model(data)
        # 将预测结果转化为类别
        pred = output.argmax(dim=1)
        print('Input: {}, Predicted class: {}'.format(data.cpu().numpy()[0], pred.cpu().numpy()[0]))
