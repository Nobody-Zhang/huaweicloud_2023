import random


def generate_0s() -> str:
    str_len = random.randint(10, 90)
    mix_len = random.randint(0, min(30, str_len // 3))
    string = "0" * (str_len - mix_len)
    for cut in range(random.randint(1, 5)):
        cut_point = random.randint(0, len(string) - 1)
        if mix_len <= 0:
            break
        cut_len = max(random.randint(1, mix_len), 18)
        mix_len -= cut_len
        string = string[:cut_point] + cut_len * str(random.randint(0, 4)) + string[cut_point:]
    return string


def generate_others(digit: int) -> str:
    str_len = random.randint(10, 90)
    mix_len = random.randint(0, min(30, str_len // 3))
    digit_len = random.randint(0,10)
    string = "0" * (str_len - mix_len)
    for cut in range(random.randint(1, 5)):
        cut_point = random.randint(0, len(string) - 1)
        if mix_len <= 0:
            break
        cut_len = max(random.randint(1, mix_len), 18)
        mix_len -= cut_len
        string = string[:cut_point] + cut_len * str(random.randint(0, 4)) + string[cut_point:]
    return string
    return "0"


if __name__ == "__main__":
    """
    with open("data/20030519/formatted_data/2.in") as f:
        max_len = 0
        min_len = 2000
        for line in f:
            mix_len = line.count("2")
            max_len = mix_len if mix_len > max_len else max_len
            min_len = mix_len if mix_len < min_len else min_len
        print(max_len, min_len)
    """
    with open("data/20030519/formatted_data/0.in", "a") as f:
        for i in range(1000):
            f.write(generate_0s() + "\n")
