# import argparse
import os
import sys
from pathlib import Path
import math
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import *
from utils.torch_utils import select_device, time_sync


# write by llr
# transform xyxy loacationn to xywh loacation, scale in (0, 1)
def xyxy2xywh(xmin: int, ymin: int, xmax: int, ymax: int, wide: int, height: int) -> tuple:
    """
    tranform xyxy location to xywh location

    :param xmin: xmin
    :param ymin: ymin
    :param xmax: xmax
    :param ymax: ymax
    :param wide: wide
    :param height: height
    :return: tuple(x,y,w,h)
    """
    x = ((xmin + xmax) // 2) / wide
    y = ((ymin + ymax) // 2) / height
    w = (xmax - xmin) / wide
    h = (ymax - ymin) / height
    return x, y, w, h



def Sliding_Window(total_status, fps, im_lis):
    # fps: real fps
    pass
    """
    single_window_cnt = [0, 0, 0, 0, 0]
    cnt_status = [0, 0, 0, 0, 0]
    threshold = 3  # More than 3 * fps is considered abnormal
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
    return max_status # Find the status with the most occurrences
    """


class YOLO_Status:
    def __init__(self):
        self.cls_ = {"close_eye": 0, "close_mouth": 1, "face": 2, "open_eye": 3, "open_mouth": 4, "phone": 5,
                     "sideface": 6}
        self.status_prior = {"normal": 0, "closeeye": 1, "yawn": 3, "calling": 4, "turning": 2}
        self.condition = [0, 1, 4, 2, 3]

    def determin(self, img, dets) -> int:
        """
        determin which status this frame belongs to\n
        0 -> normal status\n
        1 -> close eye\n
        2 -> yawn\n
        3 -> calling\n
        4 -> turning around\n

        :param img: input image, format the same as detect function
        :param dets: the detect boxes
        :returns: an int status symbol
        """
        wide, height = img.shape[1], img.shape[0]  # 输入图片宽、高
        status = 0  # 最终状态，默认为0
        driver = (0, 0, 0, 0)  # 司机正脸xywh坐标
        driver_xyxy = (0, 0, 0, 0)  # 司机正脸xyxy坐标
        driver_conf = 0  # 正脸可信度
        sideface = (0, 0, 0, 0)  # 司机侧脸xywh坐标
        sideface_xyxy = (0, 0, 0, 0)  # 侧脸xyxy坐标
        sideface_conf = 0  # 侧脸可信度
        face = (0, 0, 0, 0)  # 司机的脸，不管正侧
        face_xyxy = (0, 0, 0, 0)  # 司机的脸xyxy坐标
        phone = (0, 0, 0, 0)  # 手机xywh坐标
        openeye = (0, 0, 0, 0)  # 睁眼xywh坐标
        closeeye = (0, 0, 0, 0)  # 闭眼xywh坐标， 以防两只眼睛识别不一样
        openeye_score = 0  # 睁眼可信度
        closeeye_score = 0  # 闭眼可信度
        eyes = []  # 第一遍扫描眼睛列表
        mouth = (0, 0, 0, 0)  # 嘴xywh坐标
        mouth_status = 0  # 嘴状态，0 为闭， 1为张
        mouths = []  # 第一遍扫描嘴列表
        phone_flag = False
        face_flag = False

        # 处理boxes
        bboxes = dets
        for box in bboxes:  # 遍历每个box
            xyxy = tuple(box[:4])  # xyxy坐标
            xywh = xyxy2xywh(*xyxy, wide, height)  # xywh坐标
            conf = box[4]  # 可信度
            cls = box[5]  # 类别
            if cls == self.cls_["face"]:  # 正脸
                if .5 < xywh[0] and xywh[1] > driver[1]:
                    # box中心在右侧0.5 并且 在司机下侧
                    driver = xywh  # 替换司机
                    driver_xyxy = xyxy
                    driver_conf = conf
                    face_flag = True
            elif cls == self.cls_["sideface"]:  # 侧脸
                if .5 < xywh[0] and xywh[1] > sideface[1]:  # box位置，与face一致
                    sideface = xywh  # 替换侧脸
                    sideface_xyxy = xyxy
                    sideface_conf = conf
                    face_flag = True
            elif cls == self.cls_["phone"]:  # 手机
                if .4 < xywh[0] and .2 < xywh[1] and xywh[1] > phone[1] and xywh[0] > phone[0]:
                    # box位置在右0.4, 下0.2, 原手机右下
                    phone = xywh  # 替换手机
                    phone_flag = True  # 表示当前其实有手机
            elif cls == self.cls_["open_eye"] or cls == self.cls_["close_eye"]:  # 眼睛，先存着
                eyes.append((cls, xywh, conf))
            elif cls == self.cls_["open_mouth"] or cls == self.cls_["close_mouth"]:  # 嘴，先存着
                mouths.append((cls, xywh))

        if not face_flag:  # 没有检测到脸
            return 4 # 4 -> turning around



        # 判断状态
        face = driver
        face_xyxy = driver_xyxy
        if abs(driver[0] - sideface[0]) < .1 and abs(driver[1] - sideface[1]) < .1:  # 正脸与侧脸很接近，说明同时检测出了正脸和侧脸
            if driver_conf > sideface_conf:  # 正脸可信度更高
                status = max(status, self.status_prior["normal"])
                face = driver
                face_xyxy = driver_xyxy
            else:  # 侧脸可信度更高
                status = max(status, self.status_prior["turning"])
                face = sideface
                face_xyxy = sideface_xyxy
        elif sideface[0] > driver[0]:  # 正侧脸不重合，并且侧脸在正脸右侧，说明司机是侧脸
            status = max(status, self.status_prior["turning"])
            face = sideface
            face_xyxy = sideface_xyxy

        if face[2] == 0:  # 司机躲猫猫捏
            status = max(status, self.status_prior["turning"])

        if abs(face[0] - phone[0]) < .3 and abs(face[1] - phone[1]) < .3 and phone_flag:
            status = max(status, self.status_prior["calling"])  # 判断状态为打电话

        for eye_i in eyes:
            if eye_i[1][0] < face_xyxy[0] / wide or eye_i[1][0] > face_xyxy[2] / wide or eye_i[1][1] < face_xyxy[
                1] / height or eye_i[1][1] > face_xyxy[3] / height:
                continue
            if eye_i[0] == self.cls_["open_eye"]:  # 睁眼
                if eye_i[1][0] > openeye[0]:  # 找最右边的，下面的同理
                    openeye = eye_i[1]
                    openeye_score = eye_i[2]
            elif eye_i[0] == self.cls_["close_eye"]:  # 睁眼
                if eye_i[1][0] > closeeye[0]:  # 找最右边的，下面的同理
                    closeeye = eye_i[1]
                    closeeye_score = eye_i[2]

        for mouth_i in mouths:
            if mouth_i[1][0] < face_xyxy[0] / wide or mouth_i[1][0] > face_xyxy[2] / wide or mouth_i[1][1] < face_xyxy[
                1] / height or mouth_i[1][1] > face_xyxy[3] / height:
                continue
            if mouth_i[0] == self.cls_["open_mouth"]:  # 张嘴
                if mouth_i[1][0] > mouth[0]:
                    mouth = mouth_i[1]
                    mouth_status = 1
            elif mouth_i[0] == self.cls_["close_mouth"]:  # 闭嘴
                if mouth_i[1][0] > mouth[0]:
                    mouth = mouth_i[1]
                    mouth_status = 0

        if mouth_status == 1:  # 嘴是张着的
            status = max(status, self.status_prior["yawn"])

        if abs(closeeye[0] - openeye[0]) < .2:  # 睁眼和闭眼离得很近， 说明是同一个人两只眼睛判断得不一样
            if closeeye_score > openeye_score:  # 闭眼可信度比睁眼高
                status = max(status, self.status_prior["closeeye"])
            else:
                status = max(status, self.status_prior["normal"])
        else:  # 说明是两个人的眼睛，靠右边的是司机的眼睛
            if closeeye[0] > openeye[0]:  # 司机是闭眼
                status = max(status, self.status_prior["closeeye"])
            else:  # 司机是睁眼
                status = max(status, self.status_prior["normal"])

        return self.condition[status]




@torch.no_grad()
def yolo_run(weights=ROOT / 'yolov5s_best_openvino_model_supple_quantization_FP16/best.xml',  # model.pt path(s)
             source='',  # file/dir/URL/glob, 0 for webcam
             data=ROOT / 'yolov5s_best_openvino_model_supple_quantization_FP16/best.yaml',  # dataset.yaml path
             imgsz=(640, 640),  # inference size (height, width)
             conf_thres=0.20,  # confidence threshold
             iou_thres=0.40,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             classes=None,  # filter by class: --class 0, or --class 0 2 3
             agnostic_nms=False,  # class-agnostic NMS
             augment=False,  # augmented inference
             visualize=False,  # visualize features
             half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             FRAME_PER_SECOND=1,  # goal FPS
             window_size=3  # sliding window size
             ):
    source = str(source)

    # ------------------------- Init model -------------------------
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    fps = dataset.cap.get(cv2.CAP_PROP_FPS)
    FRAME_GROUP = int(fps / FRAME_PER_SECOND)
    # fps = FRAME_PER_SECOND
    cntt = -1
    tot_status = []
    im_lis = [] # save every frame
    YOLO_determin = YOLO_Status()

    # ------------------------- Run inference -------------------------
    t_start = time_sync()  # Start_time
    for path, im, im0s, vid_cap, s in dataset: #
        cntt += 1
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        im_lis.append((im, im0s)) # save every frame

        if cntt % FRAME_GROUP != 0:
            continue  # Skip some frames
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # print(det.numpy())
                cur_status = YOLO_determin.determin(im0, det.numpy())
                tot_status.append(cur_status)
                # cv2.imshow(str(cur_status), im0)
                # cv2.waitKey(1000)
            else:
                # Nothing detected, assume the status if "turning"
                cur_status = 4
                tot_status.append(cur_status)


    # ------------------- Attention! tot_status be like [0, 0, 2, ...] type: int--------------------------
    # for i in range(5):  # Just in case, time of the vidio isn't enouth, append 0
    #     tot_status.append(0)
    print(tot_status)
    # Post process, using the sliding window algorithm to judge the final status

    def f(probe_im_0):
        # ----------------
        im = probe_im_0[0]
        im0s = probe_im_0[1]

        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        sta = 0
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # print(det.numpy())
                sta = YOLO_determin.determin(im0, det.numpy())
            else:
                # Nothing detected, assume the status if "turning"
                sta = 4
            # cv2.imshow(f"{sta}", im0)
            # cv2.waitKey(5000)
        return sta

        # ----------------
    res = []
    pre_status = 0  # 上个状态
    pre_i = 0  # 上个状态的起始位置
    pre_las = 0  # 上个状态的采样成功次数
    # 一秒遍历
    for i in range(len(tot_status)):
        if tot_status[i] == pre_status:
            pre_las += 1
        else:
            # 目前有跳变存在
            if pre_las >= 3 and pre_status != 0:
                probe_im_1 = im_lis[int(fps * max(pre_i - 0.5, 0))]
                probe_im_2 = im_lis[int(fps * (i - 0.5))]
                status_fir = f(probe_im_1)
                status_las = f(probe_im_2)
                if status_fir == status_las:
                    if status_fir == pre_status:# 1 1
                        res.append({"periods": [pre_i - 0.75, i - 0.25], "category": pre_status})
                    elif pre_las > 3: # 0 0
                        res.append({"periods": [pre_i - 0.25, i - 0.75], "category": pre_status})

                else: # 需要进行0.25s 抽样
                    if status_fir == pre_status: # 1 0
                        if pre_las > 3:
                            res.append({"periods": [pre_i - 0.75, i - 0.75], "category": pre_status})
                        else:
                            status_025 = f(im_lis[int(fps * (pre_i - 0.75))])
                            status_075 = f(im_lis[int(fps * (i - 0.75))])
                            if pre_status == status_075 and status_025 != pre_status: # 0 0
                                pass
                            else:
                                per = []
                                if status_025 == pre_status:
                                    per.append(pre_i - 0.875)
                                else:
                                    per.append(pre_i - 0.625)
                                if status_075 == pre_status:
                                    per.append(i - 0.625)
                                else:
                                    per.append(i - 0.875)
                                res.append({"periods": per, "category": pre_status})

                    elif status_las == pre_status: # 0 1
                        if pre_las > 3:
                            # cv2.imshow("0", probe_im_1[1])
                            # cv2.imshow("1", probe_im_2[1])
                            # cv2.waitKey(10000)
                            res.append({"periods": [pre_i - 0.25, i - 0.25], "category": pre_status})
                        else:
                            status_111 = f(im_lis[int(fps * (pre_i - 0.25))])
                            status_222 = f(im_lis[int(fps * (i - 0.25))])
                            if status_111 != status_222 and status_111 != pre_status: # 无
                                pass
                            else:
                                per = []
                                if status_111 == pre_status:
                                    per.append(pre_i - 0.375)
                                else:
                                    per.append(pre_i - 0.125)
                                if status_222 == pre_status:
                                    per.append(i - 0.125)
                                else:
                                    per.append(i - 0.375)
                                res.append({"periods": per, "category": pre_status})
                            # if status_111 == status_222 and status_111 == pre_status: # 有
                            #     res.append({"periods": [pre_i - 0.375, i - 0.125], "category": pre_status})
                            # else:
                            #     pass
                    else: # 0 0
                        if pre_las > 3:
                            res.append({"periods": [pre_i - 0.25, i - 0.75], "category": pre_status})
            pre_i = i
            pre_las = 1
            pre_status = tot_status[i]
    if res[0]["periods"][0] < 0:
        res[0]["periods"][0] = 0
        if res[0]["periods"][1] - 0 < 3:
            del res[0] # 防止第一个状态为0
    for i in range(len(res)):
        res[i]["periods"][0] = int(res[i]["periods"][0] * 1000)
        res[i]["periods"][1] = int(res[i]["periods"][1] * 1000)
    # # If the count of the "phone" is more than 1.5 times of the fps, but the category is not 3, then the category is "normal"
    # cnt3 = 0
    # for i in tot_status:
    #     if i == 3:
    #         cnt3 += 1
    # if cnt3 >= 1.5 * fps and category != 3:
    #     category = 0
    # -------------------- Suit the output format --------------------
    t_end = time_sync()  # End_time
    duration = t_end - t_start

    result = {"result": {"duration": 6000, "drowsy": 0}}
    result['result']['drowsy'] = res

    result['result']['duration'] = int(duration * 1000)
    return result

if __name__ == "__main__":
      list = yolo_run(source=ROOT / '3432412.mp4')
      print(list)