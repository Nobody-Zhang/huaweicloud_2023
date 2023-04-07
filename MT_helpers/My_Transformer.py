import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from matplotlib.pylab import plt


# Transformer Classifier
class TransformerClassifier(nn.Module):
    """
    A PyTorch neural network module that implements a Transformer-based text classifier.

    Args:
        vocab_size (int): The size of the vocabulary used for the input text.
        hidden_size (int): The size of the hidden layer of the Transformer.
        num_classes (int): The number of output classes.
        num_layers (int, optional): The number of layers in the Transformer encoder. Defaults to 2.
        num_heads (int, optional): The number of attention heads in the Transformer. Defaults to 8.
        dropout (float, optional): The dropout rate used in the Transformer. Defaults to 0.1.

    Attributes:
        embedding (nn.Embedding): The embedding layer that converts the input sequence to a hidden representation.
        transformer (nn.TransformerEncoder): The Transformer encoder that processes the input sequence.
        fc (nn.Linear): The fully connected layer that produces the output logits.

    Methods:
        forward(x): Performs a forward pass of the input x through the network.

    """

    def __init__(self, vocab_size, hidden_size, num_classes, num_layers=2, num_heads=8, dropout=0.1):
        """
        Initializes a new instance of the TransformerClassifier class.

        Args:
            vocab_size (int): The size of the vocabulary used for the input text.
            hidden_size (int): The size of the hidden layer of the Transformer.
            num_classes (int): The number of output classes.
            num_layers (int, optional): The number of layers in the Transformer encoder. Defaults to 2.
            num_heads (int, optional): The number of attention heads in the Transformer. Defaults to 8.
            dropout (float, optional): The dropout rate used in the Transformer. Defaults to 0.1.
        """
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Performs a forward pass of the input x through the network.

        Args:
            x (torch.Tensor): A tensor of shape (seq_length, batch_size) representing the input text.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_classes) representing the output logits.
        """
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1, :, :]
        x = self.fc(x)
        return x


class TextDataset(data.Dataset):
    """
    A PyTorch dataset for loading text classification data from a directory.
    Assumes the data is split into 5 files, named as '0.in', '1.in', '2.in', '3.in', and '4.in',
    and each line in the file is a text sequence to classify.
    """

    def __init__(self, data_dir):
        """
        Initializes the dataset with the data in the given directory.
        Args:
            data_dir (str): The path to the directory containing the text classification data.
        """
        self.data = []
        for i in range(5):
            filename = data_dir + str(i) + '.in'
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    line = self.pad_sequence_str(line)
                    self.data.append((line, i))

    def __len__(self):
        """
        Returns the number of data samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a data sample from the dataset at the given index.
        Args:
            index (int): The index of the data sample to return.
        Returns:
            x (Tensor): A tensor of input sequence data.
            y (int): An integer label indicating the class of the input sequence.
        """
        x, y = self.data[index]
        x = [int(c) for c in x]
        x = torch.tensor(x, dtype=torch.long)
        return x, y

    def pad_sequence_str(self, seq, max_length=500, pad_char='0'):
        """
        Pads a given string sequence with a specified padding character to a maximum length.

        Args:
            seq (str): The string sequence to pad.
            max_length (int): The maximum length to pad the sequence to (default: 500).
            pad_char (str): The character to use for padding (default: '0').

        Returns:
            str: The padded string sequence.
        """
        padded_seq = seq + (pad_char * (max_length - len(seq))) if len(seq) < max_length else seq[:max_length]
        return padded_seq


class Transform:
    def __init__(self, model):
        self.model = model
        self.train_loss_list = []

    def train(self, train_loader, val_loader, optimizer, criterion, device, num_epochs=100):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_iteration(train_loader, optimizer, criterion, device)
            self.train_loss_list.append(train_loss)
            val_loss, val_acc = self.validate_iteration(val_loader, criterion, device)
            print(
                'Training...Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, '
                'Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch + 1, num_epochs, train_loss,
                                                           train_acc, val_loss, val_acc))

    def train_iteration(self, train_loader, optimizer, criterion, device):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == targets).sum().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        return train_loss, train_acc

    def validate_iteration(self, val_loader, criterion, device):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == targets).sum().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        return val_loss, val_acc

    def save_training_loss(self, loss_filepath="./training_loss.txt"):
        with open(loss_filepath, 'w') as fp:
            fp.write(' '.join((str(item) for item in self.train_loss_list)))

    def plot_training_loss(self, loss_filepath="", plot_filepath="", plot_show=True):
        training_loss_list = self.train_loss_list
        if os.path.exists(loss_filepath):
            with open(loss_filepath, 'r') as fp:
                training_loss_list = [float(x) for x in fp.read().split()]
        epochs = range(len(training_loss_list))
        plt.plot(epochs, training_loss_list, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if os.path.exists(plot_filepath):
            plt.savefig(plot_filepath)
        if plot_show:
            plt.show()

    def save_model(self, model_path="./transformer_ag_model.pth"):
        torch.save(model.state_dict(), model_path)

    def load_model(self, model_path="./transformer_ag_model.pth"):
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
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

    transformer = Transform(model)
    transformer.train(train_loader, val_loader, optimizer, criterion, device, num_epochs)
    # transformer.save_model()
    transformer.save_training_loss("test_training_loss_file.txt")
    transformer.plot_training_loss("test_training_loss_file.txt")
