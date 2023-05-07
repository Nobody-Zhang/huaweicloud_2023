import os
from matplotlib.pylab import plt


class SlidingWindow:
    @staticmethod
    def switch_window(sequence: str, fps=3, window_size=1, detect_second=3, num_classes=5, tolerance=0) -> int:
        # window queue for detection
        window = [int(seq) for seq in sequence[:window_size]]
        latent_occurrence = [0] * num_classes
        latent_occurrence[max(set(window), key=window.count)] += 1

        # slide
        for s in range(len(sequence) - window_size):
            window.append(int(sequence[s + len(window)]))
            window.pop(0)
            latent_occurrence[max(set(window), key=window.count)] += 1

        # output detection, 2 > 1
        if latent_occurrence[2] >= fps * detect_second - tolerance:
            return 2
        max_occur = 0
        max_num = 0
        for i in [1, 3, 4]:
            if latent_occurrence[i] >= max(fps * detect_second - tolerance, max_occur):
                max_occur = latent_occurrence[i]
                max_num = i
        return max_num


def window_dataset_detection(path, frames=30, tolerant=20):
    correct_N_total = []
    for filename in os.listdir(path):
        tot_input = 0
        correct_output = 0
        file_path = os.path.join(path, filename)
        actual_label = int(filename[0])
        with open(file_path) as f_in:
            for line in f_in:
                tot_input += 1
                detected_label = SlidingWindow.switch_window(line.strip(), fps=frames, tolerance=tolerant)
                correct_output = correct_output + 1 if actual_label == detected_label else correct_output
                if actual_label != detected_label:
                    print(f"{actual_label} {detected_label} {line}")
            correct_N_total.append((correct_output, tot_input))
        f_in.close()
    return correct_N_total


if __name__ == "__main__":
    dataset_path = "transformer_data_orig"
    frames = 30
    """
    dataset_path = "transformer_data_extracted"
    frames = 3
    """

    """
    accuracy_curve = [0] * 20
    for tolerant_test_num in range(20):
        detection = window_dataset_detection(dataset_path, frames=frames, tolerant=tolerant_test_num)
        tot_input = 0
        tot_accurate = 0
        for (i, j) in detection:
            tot_accurate += i
            tot_input += j
        accuracy_curve[tolerant_test_num] = tot_accurate / tot_inputS

    plt.plot([i for i in range(20)], accuracy_curve, label='Accuracy by Tolerance')
    plt.title('Accuracy rate')
    plt.xlabel('Tolerance')
    plt.ylabel('Accuracy')
    plt.show()
    """
    detection = window_dataset_detection(dataset_path, frames=frames, tolerant=10)
    tot_input = 0
    tot_accurate = 0
    for (i, j) in detection:
        tot_accurate += i
        tot_input += j
    print(tot_accurate, tot_input)
