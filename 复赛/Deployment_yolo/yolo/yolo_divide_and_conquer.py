# import argparse
import psutil
import os
import sys
import time
from pathlib import Path
# import gc
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


# 将字节转换为GB
def bytes_to_gigabytes(bytes_value):
    return bytes_value / (1024 * 1024 * 1024)

def load_imgs(dataset, half, device):
    il = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        il.append((im, im0s))  # save every frame
    return il


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
        :param dets: to detect boxes
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
                    # print("phone and config: ", phone, conf)
            elif cls == self.cls_["open_eye"] or cls == self.cls_["close_eye"]:  # 眼睛，先存着
                eyes.append((cls, xywh, conf))
            elif cls == self.cls_["open_mouth"] or cls == self.cls_["close_mouth"]:  # 嘴，先存着
                mouths.append((cls, xywh))

        if not face_flag:  # 没有检测到脸
            return 4  # 4 -> turning around

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

        # if self.condition[status] == 0:
        #     print("status: ", status)
        #     print("openeye: ", openeye_score)
        #     print("closeeye: ", closeeye_score)

        return self.condition[status]


@torch.no_grad()
def yolo_run(weights=ROOT / 'fine_tune_openvino_model/best.xml',  # model.pt path(s)
             source='',  # file/dir/URL/glob, 0 for webcam
             data=ROOT / 'fine_tune_openvino_model/best.yaml',  # dataset.yaml path
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
             frame_per_second=2,  # 分治的中间向左右的帧率
             iou_presice_b_search=0.05  # 二分时间误差系数，准确率优先，给到0.05
             ):
    source = str(source)
    # print("algo: ", )
    # print("weights: ", weights)
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
    # print("fps: ", fps)
    FRAME_GROUP = int(fps / frame_per_second)
    # print("FRAME_GROUP: ", FRAME_GROUP)
    # fps = FRAME_PER_SECOND
    cntt = -1
    im_lis = load_imgs(dataset, half, device)  # 保存所有的帧便于后续分治
    tmp = []
    sta_tmp = {}

    YOLO_determin = YOLO_Status()

    def f(probe_im_0):
        # 得到probe_im_0的状态（均以帧为单位）
        if probe_im_0 in sta_tmp:
            return sta_tmp[probe_im_0]
        if probe_im_0 >= len(im_lis) or probe_im_0 < 0:
            return 0
        im = im_lis[probe_im_0][0]
        im0s = im_lis[probe_im_0][1]

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
            # if sta == 1:
            #     cv2.imshow(f"{sta}", im0)
            #     cv2.waitKey(1000)
        sta_tmp[probe_im_0] = sta
        return sta

    def b_search(l1, r1, l2, r2, n, goal_n, k, is_3=False):
        """
        注意：输入的所有的单位为帧数！
        l1: left bound1 左区间的左边界
        r1: right bound1 左区间的右边界
        l2: left bound2 右区间的左边界
        r2: right bound2 右区间的右边界
        n: 现有可能产生的误差之和
        goal_n: 目标可能产生的误差之和
        k: the target status to find 目标状态
        """
        if n <= goal_n:
            # if is_3: # 递归到这个地方，一直都是01, 10, 10, 01这样的，无法更加精确的判断结果到底的状态，一律返回真
            lef_frame_ans = max((l1 + r1) / 2, 0)
            rig_frame_ans = min((l2 + r2) / 2, len(im_lis) - 1)

            if rig_frame_ans - lef_frame_ans < 3 * fps:
                # print(f"False(out of range): status: {k}\n l1: {l1}, r1: {r1}, l2: {l2}, r2: {r2}")
                return [False]  # 可能出现的边界条件的判断
            # print(f"True: status: {k}\n l1: {l1}, r1: {r1}, l2: {l2}, r2: {r2}")
            return [True, lef_frame_ans / fps, rig_frame_ans / fps]  # 表示可行，并且返回边界的值

        mid1 = int((l1 + r1) / 2)
        mid2 = int((l2 + r2) / 2)
        sta1 = f(mid1)
        sta2 = f(mid2)
        # print(f"\n l1: {l1} r1: {r1} l2: {l2} r2: {r2}")
        # print(f"mid1: {mid1}, mid2: {mid2}, sta1: {sta1}, sta2: {sta2}, goal:{k}")

        # 1 1
        if sta1 == k and sta2 == k:
            return b_search(l1, mid1, mid2, r2, n / 2, iou_presice_b_search * (mid2 - mid1) / fps,
                            k)  # 就算是需要判断的3s，无论如何都是可行的

        # 0 0
        if sta1 != k and sta2 != k:
            if is_3:
                # print(f"False(0, 0): status: {k}\n l1: {l1}, r1: {r1}, l2: {l2}, r2: {r2}")
                return [False]  # 如果是需要判断的3s，则无论如何都是不可行的
            return b_search(mid1, r1, l2, mid2, n / 2, iou_presice_b_search * (l2 - r1) / fps, k)  # 继续搜索边界，提升精度

        # 1 0
        if sta1 == k and sta2 != k:
            if is_3:  # 固定时长，多迭代一轮
                return b_search(l1, mid1, l2, mid2, n / 2, iou_presice_b_search * 3 * 0.25, k, True)
            return b_search(l1, mid1, l2, mid2, n / 2, iou_presice_b_search * (l2 - mid1) / fps, k)

        # 0 1
        if sta1 != k and sta2 == k:
            if is_3:
                return b_search(mid1, r1, mid2, r2, n / 2, iou_presice_b_search * 3 * 0.25, k, True)
            return b_search(mid1, r1, mid2, r2, n / 2, iou_presice_b_search * (mid2 - r1) / fps, k)

    def divide_and_conquer(l, r):
        # 分治算法，l和r表示的是左右的边界, [l, r]，且左右的状态和l - 0.5 * fps, r + 0.5 * fps的状态不一样
        # print(f"l: {l}, r: {r}")
        if r - l < 3 * fps:  # 区间小于3s
            return
        mid = int((l + r) / 2)  # 选中间的帧
        sta_mid = f(mid)
        i = 1
        j = 1
        if sta_mid != 0:
            while int(mid - 0.375 * i * fps) >= l and f(int(mid - 0.375 * i * fps)) == sta_mid:
                i += 1
            while int(mid + 0.375 * j * fps) <= r and f(int(mid + 0.375 * j * fps)) == sta_mid:
                j += 1
            if i + j >= 9:  # 表示当前已经有2.625s，但是需要更进一步二分判断
                # 注意保存的是l1，r2的帧，因为这俩都判断是不可行的
                tmp.append([i + j == 9, int(mid - 0.375 * i * fps), int(mid + 0.375 * j * fps), sta_mid])
        divide_and_conquer(l, int(mid - fps * i * 0.375))
        divide_and_conquer(int(mid + fps * j * 0.375), r)
        return

    # ------------------------- Run inference -------------------------
    t_start = time_sync()  # Start_time
    divide_and_conquer(0, len(im_lis) - 1)
    # ------------------- Attention! tot_status be like [0, 0, 2, ...] type: int--------------------------
    # for i in range(5):  # Just in case, time of the vidio isn't enouth, append 0
    #     tot_status.append(0)
    # tot_status.append(0)  # 为了最后一个状态的判断，需要多加一个0
    # print(tot_status)
    # Post process, using the sliding window algorithm to judge the final status
    res = []
    pre_i = 0  # 上个状态的起始帧（抽帧之后的 -----> FRAME_PER_SECOND）
    # 每一帧（抽帧之后的）遍历
    tmp.sort(key=lambda x: x[1])
    # print(tmp)
    for i in tmp:
        min_t = (i[2] - i[1]) / fps - 0.75
        _ = b_search(i[1], i[1] + fps * 0.375, i[2] - fps * 0.375, i[2], 0.375, min_t * iou_presice_b_search, i[3], is_3=i[0])
        if _[0]:  # 表示当前出现了大于3s的
            res.append({"periods": [int(_[1] * 1000), int(_[2] * 1000)], "category": i[3]})
    # -------------------- Suit the output format --------------------
    t_end = time_sync()  # End_time
    duration = t_end - t_start

    result = {"result": {"duration": 6000, "drowsy": 0}}
    result['result']['drowsy'] = res

    result['result']['duration'] = int(duration * 1000)
    return result


if __name__ == "__main__":
    list = yolo_run(source=ROOT / 'zipped.mp4')
    print(list)