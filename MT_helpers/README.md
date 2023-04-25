# Transformer
We utilize the Transformer under nn.module as our classifier, and format input using the Dataset module under torch.utils.data.
## Install
1. Create a virtual environment
2. Install **pytorch with cuda** following the requirements on https://pytorch.org/get-started/locally/
3. Make sure to have the training dataset ready (this will also be the validating dataset by default)

## Model Building
1. Import necessary modules and classes
```python
from My_Transformer import TransformerClassifier
```

2. 6 parameters are taken by the Transformer classifier, by which 4 are required.
```python
vocab_size = 5  # The size of the vocabulary used for the input text.
hidden_size = 32  # The size of the hidden layer of the Transformer.
num_classes = 5  # The number of output classes.
num_layers = 2  # The number of layers in the Transformer encoder. Defaults to 2.
num_heads = 4  # The number of attention heads in the Transformer. Defaults to 8.
dropout = 0.1  # The dropout rate used in the Transformer. Defaults to 0.1.

model = TransformerClassifier(vocab_size, hidden_size, num_classes, num_layers, num_heads, dropout)
```

## Training
1. Import necessary modules and classes
```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from matplotlib.pylab import plt
import numpy as np
from My_Transformer import TextDataset, Transform
```
2. Setup training and validating dataset
```python
batch_size = 32
dataset_path = 'RNN_Generated_Training/'  # using same dataset for training and validating 

train_dataset = TextDataset(dataset_path)
val_dataset = TextDataset(dataset_path)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
```
3. Initialize the transformer user (Transform)
```python
transformer = Transform(model)
```
4. Start training
```python
lr = 0.001  # learning rate
num_epochs = 10  # training times

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

transformer.train(dataset_path=dataset_path, num_epochs=num_epochs, max_seq_length=50)
```

## Other
1. Save and load model
```python
# Load model before training
transformer.load_model()  # load from transformer_ag_model.pth by default

# Save model after training
transformer.save_model()  # save to transformer_ag_model.pth by default
```
2. Evaluate model
```python
transformer.evaluate('RNN_Generated_Training/', confusion_matrix=True)  # Optional: print the confusion matrix
```
3. Save and plot training loss
```python
# Save after training
transformer.save_training_loss("test_training_loss_file.txt")

# Plot at any time
transformer.plot_training_loss("test_training_loss_file.txt")
```