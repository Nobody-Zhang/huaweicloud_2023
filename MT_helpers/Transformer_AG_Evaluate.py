import sys
import torch
import torch.utils.data as data

# setting path
sys.path.append('../Transformer_Auto_Generated.py')

from Transformer_Auto_Generated import TextDataset, TransformerClassifier


def evaluate(model, eval_loader, optimizer, criterion, device):
    model.train()
    val_loss = 0.0
    val_acc = 0.0
    for inputs, targets in eval_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        val_acc += (predicted == targets).sum().item()
    val_loss /= len(eval_loader)
    val_acc /= len(eval_loader.dataset)
    return val_loss, val_acc


if __name__ == "__main__":
    device = torch.device("cpu")
    vocab_size = 5
    hidden_size = 32
    num_classes = 5
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    batch_size = 32
    lr = 0.001
    num_epochs = 100
    eval_dataset = TextDataset('../RNN_Train_in/')
    eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    model = TransformerClassifier(vocab_size, hidden_size, num_classes, num_layers, num_heads, dropout)
    model.load_state_dict(torch.load('transformer_ag_model.pth'))
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total:.2f}% ({correct} out of {total})")
