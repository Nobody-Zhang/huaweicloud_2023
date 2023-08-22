import os
import time

import yolo
import gc
import argparse
import yolo_divide_and_conquer
def read_txt(file_name):
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
                value = int(float(value.strip()))
                data[key] = value
            datas.append(data)
            # print(data)
        return datas
    except FileNotFoundError:
        return None


def iou_cal(infe_data, true_datas):
    if len(true_datas) == 0:  # 说明是一个负样本，空列表
        return 0, None
    infe_begin = infe_data.get('begin_time')
    infe_end = infe_data.get('end_time')
    iou = 0
    index = 0
    for i, true_data in enumerate(true_datas):
        true_begin = true_data.get('begin_time')
        true_end = true_data.get('end_time')
        i_begin = max(true_begin, infe_begin)
        i_end = min(true_end, infe_end)
        u_begin = min(true_begin, infe_begin)
        u_end = max(true_end, infe_end)
        cur_iou = (i_end - i_begin) / (u_end - u_begin)
        if cur_iou > iou:
            index = i
            iou = cur_iou
    infe_kind = infe_data.get('kind')
    true_kind = true_datas[index].get('kind')
    if infe_kind == true_kind:
        return iou, None
    else:
        return iou, true_kind


def judge(txt_name, infe_datas):
    iou_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    if os.path.getsize(txt_name) == 0:  # 说明这是一个负样本/空文件
        true_datas = []
    else:
        true_datas = read_txt(txt_name)
    print('\n================validation=================')
    print(true_datas)
    print("==========================================")
    # 把单位统一变成毫秒,如果单位是秒的话就需要，将注释去掉即可
    # if len(true_datas) != 0 and true_datas[len(true_datas) - 1]['begin_time'] // 1000 <= 0:
    #     for i in range(len(true_datas)):
    #         true_datas[i]['begin_time'] *= 1000
    #         true_datas[i]['end_time'] *= 1000

    # if len(infe_datas) != 0 and infe_datas[len(infe_datas) - 1]['begin_time'] // 1000 <= 0:
    #     for i in range(len(infe_datas)):
    #         infe_datas[i]['begin_time'] *= 1000
    #         infe_datas[i]['end_time'] *= 1000
    # 考虑到可能是一个负样本的情况
    if len(true_datas) == 0:
        if len(infe_datas) == 0:
            return 1.0  # 成功判断出是负样本，满分！
        else:  # 明明是负样本缺还给出了一些推理结果，死定了
            return 0

    for i in range(len(true_datas)):
    # 剔除间隔小于3秒的情况
        if true_datas[i]['end_time'] - true_datas[i]['begin_time'] < 3000:
            true_datas.pop(i)
    if len(true_datas) == 0:
        return -1 # 标注错误

    # 正样本但是没识别出来
    if len(infe_datas) == 0:
        return 0

    correct = 0
    f1_scores = []
    # 正常样本的情况
    for k in iou_range:
        correct = 0
        for infe_data in infe_datas:
            cur_iou, true_kind = iou_cal(infe_data, true_datas)
            if true_kind is None and cur_iou != 0:  # 说明至少类型推断正确了
                if cur_iou > k:
                    correct += 1
        P = correct / len(infe_datas)
        R = correct / len(true_datas)
        if P == 0 or R == 0:
            f1_score = 0
        else:
            # print(P, R)
            f1_score = 2 * P * R / (P + R)
        f1_scores.append(f1_score)
    print("\n================f1_scores=================")
    print("0.5: " + str(f1_scores[0])+ ", 0.55: " + str(f1_scores[1])
          + ", 0.6: " + str(f1_scores[2])+ ", 0.65: " + str(f1_scores[3])
          + ", 0.7: " + str(f1_scores[4])+ ", 0.75: " + str(f1_scores[5])
          + ", 0.8: " + str(f1_scores[6])+ ", 0.85: " + str(f1_scores[7])
          + ", 0.9: " + str(f1_scores[8])+ ", 0.95: " + str(f1_scores[9]))
    f1_scores_average = sum(f1_scores) / 10

    if f1_scores_average <= 0.8:
        with open(save_path, "a") as file:
            file.write(f"video_name: {txt_name[:-4]}.mp4" + "\n")
            file.write(f"f1_scores_average: {f1_scores_average}" + "\n")
            file.write("infe_datas:" + "\n")
            for item in infe_datas:
                file.write(str(item) + "\n")
            file.write("true_datas:" + "\n")
            for item in true_datas:
                file.write(str(item) + "\n")
            file.write("==========================================" + "\n\n")
    return f1_scores_average


def cal_f1score_single(video_name, txt_name):
    infer_data = yolo.yolo_run(source=video_name,device=0)
    infer_data = infer_data['result']['drowsy']
    res = []
    for _ in infer_data:
        res.append(
            {'kind': _["category"], 'begin_time': _["periods"][0], 'end_time': _["periods"][1]})
    print("\n================inference=================")
    print(res)
    print("==========================================")
    f1_score = judge(txt_name, res)
    print(f1_score)



# 会炸内存
def cal_f1score_dir(video_dir,txt_dir,save_path):
    F1_scores = []
    time_tot = 0
    label_wrong_cnt = 0
    for video in os.listdir(video_dir):
        if video.endswith('.mp4'):
            print(video)
            txt_name = txt_dir + video[:-4] + '.txt'
            # print(txt_name)
            if not os.path.exists(txt_name):
                print(video+' , valid txt file not exist!')
                continue
            gc.collect()
            infer_data = yolo_divide_and_conquer.yolo_run(source=video_dir + video)
            res = []
            time_tot += infer_data['result']['duration']
            for _ in infer_data['result']['drowsy']:
                res.append(
                    {'kind': _["category"], 'begin_time': _["periods"][0], 'end_time': _["periods"][1]})
            print("\n================inference=================")
            print(res)
            print("==========================================")
            print("\n================time_used and total=================")
            print(infer_data['result']['duration'], time_tot)


            f1_score = judge(txt_name, res)
            if f1_score == -1:
                label_wrong_cnt += 1
                print("LABEL WRONG!!!")
                continue
            print(video+' , f1_score: '+str(f1_score))

            F1_scores.append(f1_score)
    print("\n======================================\n")
    print("F1_scores_all: "+str(F1_scores))
    print("F1_scores_average: "+str(sum(F1_scores)/len(F1_scores)))
    print("time_use_total: "+str(time_tot))
    print("label_wrong_cnt: " + str(label_wrong_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='/home/hzkd/DATA/')
    parser.add_argument('--txt_source', type=str, default='/home/hzkd/DATA/')
    parser.add_argument('--res_save_path', type=str, default='/home/hzkd/gsm/Deployment_yolo/failed.txt')
    opt = parser.parse_args()
    video = opt.video_source
    txt = opt.txt_source
    save_path = opt.res_save_path
    # 如果video是一个文件夹
    if os.path.isdir(video) and os.path.isdir(txt):
        cal_f1score_dir(video, txt, save_path)
    elif not os.path.isdir(video) and not os.path.isdir(txt):
        cal_f1score_single(video, txt)

