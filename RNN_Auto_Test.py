import numpy as np

from RNN_Auto_Generated import StringDataset, DataLoader, TransformerClassifier
import torch
import torch.nn as nn


def collate_fn(batch):
    """Takes a batch of input sequences and labels and pads the input sequences to have the same length.
    Returns the padded input sequences and corresponding labels as a tuple."""
    inputs, labels = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs, torch.tensor(labels)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    batch_size = 64
    model = TransformerClassifier(num_classes).to(device)
    model.load_state_dict(torch.load('rnn_model.pth'))

    # Test the model
    test_dataset = StringDataset(folder="RNN_Train_in", num_samples=100, max_seq_len=400)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("".join([str(x.item()) for x in inputs[0]]), labels[0], predicted[0])

        print(f"Accuracy on test set: {100 * correct / total:.2f}% ({correct} out of {total})")


if __name__ == "__main__":
    test()
