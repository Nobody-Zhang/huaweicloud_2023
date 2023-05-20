import ast


def formatted_output(label, sign=0):
    f_out = open(f"data/20030519/formatted_data/{label}/{sign}.in", "a")
    max_len=0
    # Open the file and read the lines
    with open(f'data/20030519/orig_data/{label}/{sign}.in', 'r') as f_in:
        lines = f_in.readlines()
        # Convert each line to a list and join the elements
        for line_num in range(len(lines)):
            line = lines[line_num]
            separate_point = line.index(" ")
            fps = int(float(line[:separate_point]))
            lst = ast.literal_eval(line[separate_point + 1:])
            s = ''.join(str(x) for x in lst)

            # write 6 fps to output file
            s_len = len(s)
            s_ptr = 0
            # new_s = ""
            new_fps = 3
            new_s_list = [""] * (fps // new_fps)
            while s_ptr < s_len - fps:
                for extracted_frames in range(1, fps // new_fps):
                    exact_frame_start = s_ptr + extracted_frames * new_fps
                    for exact_frame in range(new_fps):
                        new_s_list[exact_frame] += s[exact_frame + exact_frame_start]

                    """
                    all_freq = {}
                    for frames in s[s_ptr:exact_frame]:
                        if frames in all_freq:
                            all_freq[frames] += 1
                        else:
                            all_freq[frames] = 1
                    res = max(all_freq, key=all_freq.get)
                    new_s = new_s + res
                    """
                s_ptr += fps

            for new_s in new_s_list:
                if len(new_s) > 0:
                    f_out.write(new_s + "\n")

    f_in.close()
    f_out.close()
    print(max_len)


def changed_formatted_output(label):
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

def categorize_by_pn(label):
    max_len=0
    # Open the file and read the lines
    with open(f'data/20030519/orig_data/{label}.txt', 'r') as f_in:
        lines = f_in.readlines()
        # Convert each line to a list and join the elements
        for line_num in range(1, len(lines), 2):
            line = lines[line_num]
            line_specification = lines[line_num - 1]

            if line_specification[-8] == "1":
                with open(f"data/20030519/orig_data/{label}/1.in", "a") as f_out:
                    f_out.write(line)
            else:
                with open(f"data/20030519/orig_data/{label}/0.in", "a") as f_out:
                    f_out.write(line)

            """
            if line_specification[-8] == "1":
                with open("data/20030519/formatted_data/0.in", "a") as f0:
                    f0.write(new_s + "\n")
            else:
                f_out.write(new_s + "\n")
            """

    f_in.close()
    print(max_len)


if __name__ == "__main__":
    for i in range(1, 5):
        formatted_output(i, sign=0)
        formatted_output(i, sign=1)
