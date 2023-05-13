import ast


def formatted_output(label):
    f_out = open(f"data/20230513/formatted_data/{label}.in", "a")
    # Open the file and read the lines
    with open(f'data/20230513/orig_data/{label}.txt', 'r') as f_in:
        lines = f_in.readlines()
        # Convert each line to a list and join the elements
        for line in lines[1:]:
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
                for extracted_frames in range(fps / new_fps):
                    new_s = new_s + s[s_ptr + extracted_frames * new_fps]
                s_ptr += fps
            f_out.write(new_s + "\n")

    f_in.close()
    f_out.close()


if __name__ == "__main__":
    for i in range(5):
        formatted_output(i)