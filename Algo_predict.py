import os


def load_folder(folder="RNN_Generated_Training"):
    train_data_dir = folder

    # Load the data from files
    sequences = []
    labels = []
    for label in range(5):
        filepath = os.path.join(train_data_dir, f"{label}.in")
        with open(filepath, "r") as f:
            for line in f:
                seq = [int(c) for c in line.strip()]
                sequences.append(seq)
                labels.append(label)
    return [sequences, labels]


def classfier(sequences):
    predicted_labels = []
    for seq in sequences:
        predicted_labels.append(dc_helper(seq))
    return predicted_labels


def dc_helper(seq):
    if (len(seq)) < 90:
        return 0

    occurrence_dict = {1: 0, 2: 0, 3: 0, 4: 0}
    left_idx = max(len(seq) // 2 - 90, 0)
    right_idx = min(len(seq) // 2 + 90, len(seq))

    dc_left = dc_helper(seq[:len(seq) // 2])
    dc_right = dc_helper(seq[len(seq) // 2:])

    if dc_left or dc_right:
        return dc_left or dc_right

    for i in range(left_idx, right_idx):
        if seq[i] != 0:
            occurrence_dict[seq[i]] += 1

    return get_highest_occurrence_number(occurrence_dict)


def get_highest_occurrence_number(d):
    """Helper method for getting highest occurring number in dictionary, return 0 if the occurrence is lower than 80"""
    for num, occurrence in d.items():
        if occurrence > 120:  # intended to set to 70, risking 1% inaccuracy for 97% input accuracy
            return num
    return 0


def count_same(labels, predicts):
    size = min(len(labels), len(predicts))
    same = 0
    for i in range(size):
        same += 1 if labels[i] == predicts[i] else 0
    return same


if __name__ == "__main__":
    sequ, label = load_folder()  # folder="RNN_Train_in"
    predicted = classfier(sequ)
    correct = count_same(label, predicted)
    total = len(label)
    print(f"Accuracy rate: {100 * correct / total:.2f}% ({correct} out of {total})")
