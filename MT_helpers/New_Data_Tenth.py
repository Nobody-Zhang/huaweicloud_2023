import os


def in_n_out(input_dir, output_dir, slice_n):
    for i in range(0, 5):
        input_data = open(os.path.join(input_dir, str(i) + ".in"), 'r')

        for line in input_data:
            line = line.strip()
            output_fp = open(os.path.join(output_dir, str(i) + ".in"), 'a')
            line_separated = [""] * slice_n
            for digit in range(0, len(line)):
                line_separated[digit % slice_n] += line[digit]

            for separated in line_separated:
                output_fp.write(separated + '\n')


if __name__ == "__main__":
    in_n_out("data/transformer_data_orig", "data/transformer_data_extracted", 10)
