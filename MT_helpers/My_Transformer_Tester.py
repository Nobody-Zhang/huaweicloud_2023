from My_Transformer import *

if __name__ == "__main__":
    vocab_size = 5
    hidden_size = 32
    num_classes = 5
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    model = TransformerClassifier(vocab_size, hidden_size, num_classes, num_layers, num_heads, dropout)
    device = torch.device('cpu')
    model.to(device)

    transformer = Transform(model)
    transformer.load_model("transformer_ag_model.pth")
    # transformer.train(dataset_path='RNN_Generated_Training/', num_epochs=10)
    # transformer.save_model()
    # transformer.save_training_loss("test_training_loss_file.txt")
    # transformer.plot_training_loss("test_training_loss_file.txt")
    matrix = transformer.evaluate('../RNN_Train_in/', confusion_matrix=True)
    # Create axis labels
    x_label = 'Predicted Category'

    # Print confusion matrix with axis labels and indices
    print(f'{"":10s}{x_label:20s}')
    print(f'{"-" * 45}')
    for i in range(5):
        row_str = ' '.join(f'{num:.4f}' for num in matrix[i])
        print(f'{i:4d} | {row_str} {"":4s}|')
    print(f'{"-" * 45}')
    print(f'{"":4s} {"|":4s} {"0":6s} {"1":6s} {"2":6s} {"3":6s} {"4":6s}')
