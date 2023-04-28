import os
from matplotlib.pylab import plt


class SlidingWindow():
    @staticmethod
    def detect_behavior(sequence: str, fps=3, tolerance=1, detect_second=3, num_classes=5, priority=None) -> int:
        if priority is None:
            priority = [1] * num_classes
        seq_len = len(sequence)
        detected = [False] * num_classes

        lp = 0
        rp = min(lp + detect_second * fps + 1, len(sequence), seq_len)
        occurrence = {}
        for i in range(num_classes):
            occurrence[str(i)] = 0
        for i in range(lp, rp):
            occurrence[sequence[i]] += 1
            # it matters, and it has exceeded the requirements
            if priority[int(sequence[i])] > 0 and occurrence[sequence[i]] >= detect_second * fps - tolerance:
                detected[int(sequence[i])] = True

        while rp < seq_len:
            occurrence[sequence[rp]] += 1
            # it matters, and it has exceeded the requirements
            if priority[int(sequence[rp])] > 0 and occurrence[sequence[rp]] >= detect_second * fps - tolerance:
                detected[int(sequence[rp])] = True

            # delete the left and add the right of sliding window
            occurrence[sequence[lp]] -= 1
            lp += 1
            rp += 1

        detected_value = 0
        max_priority = 0
        for i in range(num_classes):
            if detected[i] and priority[i] > max_priority:
                max_priority = priority[i]
                detected_value = i
        return detected_value

    @staticmethod
    def dataset_detection(path, frames=30, tolerant=20, priority=[0, 1, 2, 1, 1]):
        correct_N_total = []
        for filename in os.listdir(path):
            tot_input = 0
            correct_output = 0
            file_path = os.path.join(path, filename)
            actual_label = int(filename[0])
            with open(file_path) as f_in:
                for line in f_in:
                    tot_input += 1
                    detected_label = SlidingWindow.detect_behavior(line.strip(), fps=frames,
                                                                   tolerance=tolerant, priority=priority)
                    correct_output = correct_output + 1 if actual_label == detected_label else correct_output
                    """
                    print(f"actual: {actual_label}; detected: {detected_label}")
                    if actual_label != detected_label:
                        print(str(actual_label) + " " + str(detected_label) + " " + line)
                    """
                correct_N_total.append((correct_output, tot_input))
            f_in.close()
        return correct_N_total


if __name__ == "__main__":
    dataset_path = "transformer_data_orig"
    frames = 30
    tolerant = 20
    priority = [0, 1, 2, 1, 1]
    """
    dataset_path = "transformer_data_extracted"
    frames = 3
    tolerant = 0
    priority = [0, 1, 2, 1, 1]
    """
    accuracy_curve = [0] * 60
    for tolerant_test_num in range(60):
        detection = SlidingWindow.dataset_detection(dataset_path, frames=frames,
                                                    tolerant=tolerant_test_num, priority=priority)
        tot_input = 0
        tot_accurate = 0
        for (i, j) in detection:
            tot_accurate += i
            tot_input += j
        accuracy_curve[tolerant_test_num] = tot_accurate / tot_input

    plt.plot([i for i in range(60)], accuracy_curve, label='Accuracy by Tolerance')
    plt.title('Accuracy rate')
    plt.xlabel('Tolerance')
    plt.ylabel('Accuracy')
    plt.show()
