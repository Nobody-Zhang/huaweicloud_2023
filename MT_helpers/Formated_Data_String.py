import ast


def formated_output(label):
    # Open the file and read the lines
    with open(f'data/20230513/{label}.in', 'r') as f_in:
        lines = f_in.readlines()

    with open(f"data/20230513/{label}.in", "w") as f_out:
        # Convert each line to a list and join the elements
        for line in lines:
            lst = ast.literal_eval(line)
            s = ''.join(str(x) for x in lst)
            f_out.write(s + "\n")

    f_in.close()
    f_out.close()


if __name__ == "__main__":
    for i in range(5):
        formated_output(i)
