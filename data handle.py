import os

# 定义数据文件夹路径
data_dir = 'Rnn_Train_in/'

# 遍历四个数据文件，找到最长的字符串
max_length = 0
for i in range(5):
    filename = os.path.join(data_dir, str(i) + '.in')
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > max_length:
                max_length = len(line)

# 将数据填充至相同长度，存储到Train_in文件夹
output_dir = 'Train_in/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(5):
    input_filename = os.path.join(data_dir, str(i) + '.in')
    output_filename = os.path.join(output_dir, str(i) + '.in')
    with open(input_filename, 'r') as f_in, open(output_filename, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            padding = '0' * (max_length - len(line))
            line_padded = line + padding
            f_out.write(line_padded + '\n')
