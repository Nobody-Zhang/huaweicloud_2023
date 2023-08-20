import yolo
import gc
import os


def read_file_contents(file_name):
    try:
        # 打开文件并读取内容
        with open(file_name, "r") as file:
            content = file.read()
        # 拆分内容为多个块
        blocks = content.strip().split("\n\n")
        datas = []
        # 遍历每个块，提取并保存数字
        for block in blocks:
            data = {}
            lines = block.split("\n")
            for line in lines:
                key, value = line.split(":")
                key = key.strip()
                value = int(value.strip())
                data[key] = value
            datas.append(data)
            # print(data)
        return datas
    except FileNotFoundError:
        return None


def judge(txt_name, infe_data, fps):
    data_read = read_file_contents(txt_name)
    data_true = []
    iou_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    F1_scores = []
    for dict in data_read:
        kind = dict.get('kind')
        begin_time = float(dict.get('begin_time')) / 1000
        end_time = float(dict.get('end_time')) / 1000
        data_true.append({
            'kind': kind,
            'begin_time': begin_time,
            'end_time': end_time
        })
    if len(infe_data) == len(data_true):  # 说明至少从数量上来说推断是没大问题的，接下来就直接使用索引依次比较了
        for k in iou_range:  # 逐渐上升iou合格的标准
            correct = 0
            for i in range(len(infe_data)):
                if infe_data[i].get('kind') == data_true[i].get('kind'):  # 说明至少种类也推断正确了
                    true_begin_time = data_true[i].get('begin_time')
                    true_end_time = data_true[i].get('end_time')
                    infe_begin_time = infe_data[i].get('begin_time')
                    infe_end_time = infe_data[i].get('end_time')
                    i_begin_time = max(true_begin_time, infe_begin_time)
                    i_end_time = min(true_end_time, infe_end_time)
                    u_begin_time = min(true_begin_time, infe_begin_time)
                    u_end_time = max(true_end_time, infe_end_time)
                    cur_iou = (i_end_time - i_begin_time) / (u_end_time - u_begin_time)
                    if cur_iou >= k:
                        correct += 1
            P = correct / len(infe_data)
            R = correct / len(data_true)
            if P == 0 and R == 0:
                f1_score = 0
            else:
                f1_score = 2 * P * R / (P + R)
            F1_scores.append(f1_score)
        print(F1_scores)
        F1_score_avrage = sum(F1_scores) / 10
        return F1_score_avrage
    else:
        print('没有找全所有的错误行为，恐怕只能得0分了')
        return 0


def main():
    folder_path = '/home/master/zhoujian/Documents/zgb/videos_10'

    # 存储所有.mp4文件的绝对路径的列表
    mp4_files = []

    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))

    tot = []
    # 0.2 - 0.5, 0.05


    with open("output_frame_per_second=1.txt", "a") as file:
        for i in range(0, 10):
            thre = 0.2 + i * 0.05
            print("thre: ", thre)
            # 添加换行符以分隔不同的输出
            file.write("=============================================================\n")
            file.write("thre: " + str(thre) + "\n\n\n\n\n\n\n\n")

            for mp4_file in mp4_files:
                file.write("name: " + mp4_file + "\n")
                print(mp4_file)
                txt_name = mp4_file[:-4] + '.txt'
                infe_data = yolo.yolo_run(source=mp4_file, thre=thre, frame_per_second = 1)
                print("infe_data: ", infe_data)
                res = []
                for _ in infe_data['result']['drowsy']:
                    res.append(
                        {'kind': _["category"], 'begin_time': _["periods"][0] / 1000, 'end_time': _["periods"][1] / 1000})
                print(res)
                file.write("infer_data: " + str(res) + "\n")
                F1_score = judge(txt_name, res, 30)
                gc.collect()
                file.write("F1_score: " + str(F1_score) + "\n")
                print(F1_score)
                tot.append(F1_score)
                print("\n\n\n\n")
            tot_aver = sum(tot) / len(tot)
            file.write("tot_aver: " + str(tot_aver) + "\n\n\n\n")


if __name__ == '__main__':
    main()
