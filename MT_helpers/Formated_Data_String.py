import ast


def formatted_output(label):
    f_out = open(f"data/20030519/formatted_data/{label}.in", "a")
    max_len=0
    # Open the file and read the lines
    with open(f'data/20030519/orig_data/{label}.txt', 'r') as f_in:
        lines = f_in.readlines()
        # Convert each line to a list and join the elements
        for line_num in range(1, len(lines), 2):
            line = lines[line_num]
            separate_point = line.index(" ")
            fps = int(float(line[:separate_point]))
            lst = ast.literal_eval(line[separate_point + 1:])
            s = ''.join(str(x) for x in lst)

            # write 6 fps to output file
            s_len = len(s)
            s_ptr = 0
            new_s = ""
            new_fps = 6
            while s_ptr < s_len:
                for extracted_frames in range(fps // new_fps):
                    exact_frame = s_ptr + extracted_frames * new_fps
                    new_s = new_s + s[exact_frame] if exact_frame < s_len else new_s
                s_ptr += fps
            f_out.write(new_s + "\n")

    f_in.close()
    f_out.close()
    print(max_len)


if __name__ == "__main__":
    for i in range(5):
        formatted_output(i)
