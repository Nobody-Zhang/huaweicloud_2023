def Sliding_Window(total_status, fps, window_size):
    single_window_cnt = [0, 0, 0, 0, 0]
    cnt_status = [0, 0, 0, 0, 0]
    threshold = 3  # 大于3s就认为是这个状态
    for i in range(len(total_status) - int(window_size * fps)):
        if i == 0:
            for j in range(int(window_size * fps)):
                single_window_cnt[int(total_status[i + j])] += 1
        else:
            single_window_cnt[int(total_status[i + int(window_size * fps) - 1])] += 1
            single_window_cnt[int(total_status[i - 1])] -= 1
        for j in range(1, 5):
            if single_window_cnt[j] >= threshold * fps:
                cnt_status[j] += 1
    cnt_status[0] = 0
    max_status = 0
    for i in range(1, 5):
        if cnt_status[i] > cnt_status[max_status]:
            max_status = i
    return max_status


if __name__ == '__main__':
    print(Sliding_Window([1, 1, 1, 3, 3, 3, 3, 3, 1, 0, 0, 0, 0, 0], 1, 4))