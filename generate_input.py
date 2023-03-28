import os.path
import random

def generate_zero(accuracy, times = 20):
    sequences = []
    for i in range(times):
        seq_len = random.randint(300, 400)  # length of the video
        num_seq = int(accuracy * seq_len)  # number of behavior correctly recognised

        sequence = [str(0)] * num_seq + [str(random.randint(1, 4)) for j in range(seq_len - num_seq)]
        random.shuffle(sequence)
        sequences.append("".join(sequence))
    return sequences


def generate(accuracy, num=1, times=20):
    sequences = []
    for i in range(times):
        seq_len = random.randint(300, 400)  # length of the video
        recog_len = random.randint(81, 300)  # length of the behavior
        num_recog = int(accuracy * recog_len)  # number of behavior correctly recognised
        num_recog_other = recog_len - num_recog  # number of behavior wrongly recognised

        recog_seq = [str(num)] * num_recog + [str((random.randint(1, 4) + num) % 5) for j in range(num_recog_other)]
        random.shuffle(recog_seq)

        left_rest_seq_len = random.randint(0, seq_len - recog_len)
        left_zero_len = int(accuracy * left_rest_seq_len)
        left_rest_seq = ["0"] * left_zero_len + [str(random.randint(1, 4) % 5) for j in range(left_rest_seq_len
                                                                                              - left_zero_len)]
        random.shuffle(left_rest_seq)
        right_rest_seq_len = seq_len - recog_len - left_rest_seq_len
        right_zero_len = int(accuracy * right_rest_seq_len)
        right_rest_seq = ["0"] * right_zero_len + [str(random.randint(1, 4) % 5) for j in range(right_zero_len
                                                                                                - right_zero_len)]
        random.shuffle(right_rest_seq)

        sequence = left_rest_seq + recog_seq + right_rest_seq
        sequences.append("".join(sequence))
    return sequences


if __name__ == "__main__":
    accuracy = 0.9
    generate_number = 0
    times = 1000
    sequences = generate(accuracy, generate_number, times) if generate_number != 0 else generate_zero(accuracy,times)
    with open(os.path.join("RNN_Generated_Training", str(generate_number) + ".in"), "a") as f:
        for seq in sequences:
            f.write(seq + "\n")
