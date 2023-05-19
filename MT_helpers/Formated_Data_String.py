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
            line_specification = lines[line_num - 1]
            separate_point = line.index(" ")
            fps = int(float(line[:separate_point]))
            lst = ast.literal_eval(line[separate_point + 1:])
            s = ''.join(str(x) for x in lst)

            # write 6 fps to output file
            s_len = len(s)
            s_ptr = 0
            new_s = ""
            new_fps = 6
            while s_ptr < s_len - fps:
                for extracted_frames in range(1, fps // new_fps):
                    exact_frame = s_ptr + extracted_frames * new_fps
                    all_freq = {}
                    for frames in s[s_ptr:exact_frame]:
                        if frames in all_freq:
                            all_freq[frames] += 1
                        else:
                            all_freq[frames] = 1
                    res = max(all_freq, key=all_freq.get)
                    new_s = new_s + res
                s_ptr += fps
            """
            if line_specification[-8] == "1":
                with open("data/20030519/formatted_data/0.in", "a") as f0:
                    f0.write(new_s + "\n")
            else:
                f_out.write(new_s + "\n")
            """

    f_in.close()
    f_out.close()
    print(max_len)


if __name__ == "__main__":
    for i in range(5):
        formatted_output(i)
