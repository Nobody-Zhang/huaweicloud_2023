import matplotlib.pyplot as plt


class Correctness:
    def __init__(self, fps, tolerance, detect_seconds=3, num_classes=5):
        self.fps = fps
        self.tolerance = tolerance
        self.detect_seconds = detect_seconds
        self.num_classes = num_classes

    def counter(self, sequence: str) -> int:
        occurrence = [sequence.count(str(i)) for i in range(self.num_classes)]
        possible_label = [False] * self.num_classes

        for i in range(self.num_classes):
            if occurrence[i] >= self.detect_seconds * self.fps - self.tolerance:
                possible_label[i] = True

        if possible_label[2]:
            return 2
        else:
            for i in range(1, 5):
                if possible_label[i]:
                    return i

        return 0

    def algo_evaluation(self, dataset_path: str, confusion_matrix=False) -> float:
        num_sequences = 0
        num_correct = 0
        matrix = []
        for i in range(self.num_classes):
            matrix.append([0] * self.num_classes)
        for label in range(self.num_classes):
            input_fp = open(f"{dataset_path}{label}.in", "r")
            for seq in input_fp:
                prediction = self.counter(seq)
                matrix[label][prediction] += 1
                num_sequences += 1
                num_correct += 1 if label == prediction else 0

            input_fp.close()

        if confusion_matrix:
            for row in matrix:
                print(row)

        return num_correct / num_sequences


if __name__ == "__main__":
    tolerances = []
    accuracy = []
    for tol in range(5):
        tolerances.append(tol)
        correctness = Correctness(fps=6, tolerance=tol)
        rate = correctness.algo_evaluation(dataset_path="data/20230521/", confusion_matrix=True)
        accuracy.append(rate)
        print("{:.2f}%".format(rate * 100))
    plt.plot(tolerances,accuracy)
    plt.show()
