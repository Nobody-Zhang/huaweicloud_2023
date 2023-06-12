from multiprocessing import Process, Manager, Event
import time
from PIL import Image
import cv2
import argparse
import torch
import json
from torchvision import transforms
from torch import nn
from PIL import Image
from skimage.transform import resize
from model_service.pytorch_model_service import PTServingBaseService
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# import video classfication
# from Nanodet.nanodet.util import Logger, cfg1, cfg2, load_config
# from Nanodet.demo.nanodet_twostages import Predictor
# from mobilenet.model_v2_gray import MobileNetV2
from NanodetOpenvino.Nanodet import NanoDet
from MT_helpers.My_Transformer import *
# import SVM-image classfication
import svm.svmdetect as svmdetect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolo.models.common import DetectMultiBackend
from yolo.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolo.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolo.utils.plots import Annotator, colors, save_one_box
from yolo.utils.torch_utils import select_device, time_sync

from nanodet_run import *
# fps = 30
# 抽帧
FRAME_GROUP = 6
# 设置三种状态的编号
# NORMAL = 0
# EYE_CLOSE = 1
# YAWN = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model", default="nanodet", help="video classfication model name")
    parser.add_argument("--image_model", default="svm", help="image classfication model name")
    parser.add_argument("--mouth_model", default="./svm/svm_model_mouth.pkl", help="yawn classfication model name")
    parser.add_argument("--eye_model", default="./svm/svm_model_eyes.pkl", help="eye classfication model name")
    parser.add_argument("--trans_model", default="./MT_helpers/transformer_ag_model.pth", help="transformer model")
    parser.add_argument("--path", default="./day_man_001_10_1.mp4", help="path to video")
    parser.add_argument("--device", default="cpu", help="device for model use")

    return parser.parse_args()


def SVM_Handle(eye_queue, yawn_queue) -> tuple:
    eye_classifier = svmdetect.ImageClassifier("/home/ma-user/infer/model/1/svm/svm_model_eyes.pkl")
    mouth_classifier = svmdetect.ImageClassifier("/home/ma-user/infer/model/1/svm/svm_model_mouth.pkl")
    eye_gray = []
    yawn_gray = []
    # 先进行灰度处理、resize处理
    for eye_img in eye_queue:
        gray_eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye_img = cv2.resize(gray_eye_img, (60, 36))
        eye_gray.append(gray_eye_img)
    for mouth_img in yawn_queue:
        gray_mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
        gray_mouth_img = cv2.resize(gray_mouth_img, (100, 50))
        yawn_gray.append(gray_mouth_img)
    features_eyes = eye_classifier.extract_features(eye_gray)
    features_mouths = mouth_classifier.extract_features(yawn_gray)
    eye_pred, t1 = eye_classifier.classify(features_eyes)
    yawn_pred, t2 = mouth_classifier.classify(features_mouths)
    t = t1 + t2
    print(t)
    return eye_pred.tolist(), yawn_pred.tolist()
    # return Array.asList(eye_pred), yawn_pred.asList(num)


# 组合Nanodet/Yolo + SVM/MobileNetV2
class Combination:
    def __init__(self, video_model_name="nanodet", image_model_name="mobilenet",
                 mouth_model="./mobilenet/MobileNetV2_mouthclass.pth",
                 eye_model="./mobilenet/MobileNetV2_eyeclass.pth",
                 device=torch.device("cpu")):
        # 选择video classification model
        self.device = torch.device(device)
        if video_model_name == "nanodet":
            self.video_model = self.Nanodet_init()
        # elif video_model_name == "yolo":
        #      self.video_model = yolo_run

        # 选择image classification model,并创建两个线程类（对CPU影响太大，故舍弃
        """
          if image_model_name == "svm":
              self.image_eye = SVM_Eye_Process(eye_status_list,eye_queue,eyestop_event,eye_model)
              self.image_mouth = SVM_Mouth_Process(yawn_status_list,yawn_queue,yawnstop_event,mouth_model)
          elif image_model_name == "mobilenet":
              self.image_mouth = MobileNet_Yawn_Process(yawn_status_list,yawn_queue,yawnstop_event,mouth_model)
              self.image_eye = MobileNet_Eye_Process(eye_status_list,eye_queue,eyestop_event,eye_model)
          """

    def Nanodet_init(self,
                     nanodet_model1="/home/ma-user/infer/model/1/NanodetOpenvino/convert_for_two_stage/seg_face/seg_face.xml",
                     nanodet_model2="/home/ma-user/infer/model/1/NanodetOpenvino/convert_for_two_stage/face_eyes/nanodet.xml",
                     num_class1=4, num_class2=2, threshold1=0.4, threshold2=0.3):
        # current_dir = os.getcwd()
        # nanodet_model1 = os.path.join(current_dir, nanodet_model1)
        # nanodet_model2 = os.path.join(current_dir, nanodet_model2)
        # get 2 model
        nanodet_face = NanoDet(nanodet_model1, num_class1, threshold1)
        nanodet_eye_mouth = NanoDet(nanodet_model2, num_class2, threshold2)

        return [nanodet_face, nanodet_eye_mouth]

    # def Nanodet_init(self, nanodet_cfg1="./Nanodet/demo/nanodet-plus-m_416-yolo.yml",
    #                  nanodet_model1="./Nanodet/demo/model_last.ckpt",
    #                  nanodet_cfg2="./Nanodet/demo/face_eyes.yml", nanodet_model2="./Nanodet/demo/face_eyes.ckpt"):
    #
    #      local_rank = 0
    #      torch.backends.cudnn.enabled = True
    #      torch.backends.cudnn.benchmark = True
    #      load_config(cfg1, nanodet_cfg1)
    #      load_config(cfg2, nanodet_cfg2)
    #      logger = Logger(local_rank, use_tensorboard=False)
    #      # get 2 model
    #      nanodet_face = Predictor(cfg1, nanodet_model1, logger, self.device)
    #      nanodet_eye_mouth = Predictor(cfg2, nanodet_model2, logger, self.device)
    #
    #      return [nanodet_face, nanodet_eye_mouth]


# def Sliding_Window(total_status, fps, window_size):
#     single_window_cnt = [0, 0, 0, 0, 0]
#
#     threshold = 3 # 大于3帧就认为是这个状态
#     for i in range(len(total_status) - int(window_size * fps)):
#         if i == 0:
#             for j in range(int(window_size * fps)):
#                 single_window_cnt[int(total_status[i + j])] += 1
#         else:
#             single_window_cnt[int(total_status[i + int(window_size * fps) - 1])] += 1
#             single_window_cnt[int(total_status[i - 1])] -= 1
#         for j in range(1, 5):
#             if single_window_cnt[j] >= threshold*fps:
#                 return j
#     return 0


# 根据output的状态决定该图片(?视频)是哪一种状态
def SVM_Determin(eye_status, yawn_status, transform_path, tot_status: list, fps):
    output = []
    for i in range(len(eye_status)):
        # 首先判断是否打哈欠了
        if yawn_status[i] == -1:
            output.append(2)
        # 如果没有打哈欠 但是闭眼
        elif eye_status[i] == -1:
            output.append(1)
        else:
            output.append(0)
    j = 0
    for i in range(len(tot_status)):
        if tot_status[i] == -1:
            tot_status[i] = output[j]
            j = j + 1
    print(tot_status)
    # result = Transform_result(transform_path,output)
    # result = Transform_result(transform_path, tot_status)
    result = Sliding_Window(tot_status, fps, window_size = 3.6)
    # print(result[0]) # (?)
    return result


# 根据output的状态决定该图片是哪一种状态           
def Mobilenet_Determin(eye_status, yawn_status, output):
    for i in range(len(eye_status)):
        # 首先判断是否打哈欠了
        if yawn_status[i] == 1:
            output.append(2)
        # 如果没有打哈欠 但是闭眼
        elif eye_status[i] == 0:
            output.append(1)
        else:
            output.append(0)
    print(output)


def Transform_result(model_path, status_list):
    vocab_size = 5
    hidden_size = 32
    num_classes = 5
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    model = TransformerClassifier(vocab_size, hidden_size, num_classes, num_layers, num_heads, dropout)
    device = torch.device('cpu')
    model.to(device)

    transformer = Transform(model)
    transformer.load_model(model_path)

    status_str = ""
    for item in status_list:
        status_str += str(item)

    predicted = transformer.evaluate_str(status_str)
    return predicted


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

class yolo_model:
    def __init__(self,source):
        self.weights = ROOT / 'yolo/INT8_openvino_model/best_int8.xml'  # model.pt path(s)
        self.source = ROOT / str(source)  # file/dir/URL/glob, 0 for webcam
        self.data = ROOT / 'yolo/best_openvino_model/best.yaml'  # dataset.yaml path
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.20  # confidence threshold
        self.iou_thres = 0.40  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.FRAME_PER_SECOND = 1  # 改这里！！！一秒几帧
        self.window_size = 4  # 改这里！！！滑动窗口大小

    def inference(self):
        # Load model
        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if self.half else model.model.float()
        bs = 1  # batch_size
        # Dataloader

        dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt)

        vid_path, vid_writer = [None] * bs, [None] * bs

        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=self.half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        fps = dataset.cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_len = dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        FRAME_GROUP = int(fps / self.FRAME_PER_SECOND)
        fps = self.FRAME_PER_SECOND

        cntt = -1
        tot_status = []
        YOLO_determin = YOLO_Status()
        # Run inference
        t_start = time_sync()  # start_time

        # ---------------------------------------------
        for path, im, im0s, vid_cap, s in dataset:
            cntt += 1
            if cntt % FRAME_GROUP != 0:
                continue  # skip some frames
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                # s += '%gx%g ' % im.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # imc = im0.copy() if save_crop else im0  # for save_crop
                # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # print(det.numpy())
                    cur_status = YOLO_determin.determin(im0, det.numpy())
                    tot_status.append(cur_status)
                else:  # 没有检测到任何东西，当前的状态为4
                    cur_status = 4
                    tot_status.append(cur_status)

                    # print(cur_status)

            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # -------------------一定注意，这里得到的是tot_status，be like [0, 0, 2, ...]，数字！--------------------------
        for i in range(5):  # 防止视频时间不够，补0
            tot_status.append(0)

        category = Sliding_Window(tot_status, fps, self.window_size)
        print(tot_status)
        cnt3 = 0
        for i in tot_status:
            if i == 3:
                cnt3 += 1
        if cnt3 >= 1.5 * fps and category != 3:
            category = 0
        # --------------------最后的返回！！！！！！-------------------------
        t_end = time_sync()  # end_time
        duration = t_end - t_start

        result = {"result": {"category": 0, "duration": 6000}}
        result['result']['category'] = category

        # score = sigmoid(video_len / duration)

        # result['result']['duration'] = int(np.round((duration) * 1000))
        result['result']['duration'] = int(duration * 1000)
        return result

class model:
    def __init__(self):
        video_model_name = 'nanodet'
        image_model_name = 'svm'
        device = 'cpu'
        if device == 'gpu':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        # current_dir = os.getcwd()

        mouth_model_path = "/home/ma-user/infer/model/1/svm/svm_model_mouth.pkl"
        eye_model_path = "/home/ma-user/infer/model/1/svm/svm_model_eyes.pkl"

        self.Combine_model = Combination(video_model_name, image_model_name, mouth_model_path, eye_model_path, device)

    def inference(self, cap):
        # current_dir = os.getcwd()
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps / FRAME_GROUP
        # transform_path = os.path.join(current_dir, "MT_helpers/transformer_ag_model.pth")
        transform_path = "/home/ma-user/infer/model/1/MT_helpers/transformer_ag_model.pth"
        # cap = cv2.VideoCapture(video_path)
        all_start = time.time()
        cnt = 0

        # 每一帧的状态
        tot_status = []

        eye_queue = []
        yawn_queue = []

        flag = False

        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break

            # # 抽帧：取每组的第一帧
            cnt += 1
            if cnt % FRAME_GROUP != 1:
                continue

            # 识别人脸
            face_boxs = self.Combine_model.video_model[0].find_face(frame)
            face_img = face_boxs[1]
            # 判断能否裁出人脸
            if face_boxs[0] == -1 or face_boxs[0] == 0:
                tot_status.append(4)
            elif face_boxs[0] == 1:
                tot_status.append(3)
            else:
                mouth_eye_boxes = self.Combine_model.video_model[1].find_eye_mouth(face_img)
                if not mouth_eye_boxes[0] or not mouth_eye_boxes[2]:
                    tot_status.append(4)
                else:
                    eye_img = mouth_eye_boxes[3]
                    mouth_img = mouth_eye_boxes[1]
                    eye_queue.append(eye_img)
                    yawn_queue.append(mouth_img)
                    tot_status.append(-1)
                    flag = True

        print("End")

        print(f'orgin tot_status{tot_status}')

        eye_status_list = []
        yawn_status_list = []
        if(flag):
            eye_status_list, yawn_status_list = SVM_Handle(eye_queue, yawn_queue)

        print(f'eye_status_list{eye_status_list}')
        print(f'yawn_status_list{yawn_status_list}')

        category = SVM_Determin(eye_status_list, yawn_status_list, transform_path, tot_status, fps)
        all_end = time.time()
        duration = all_end - all_start
        result = {"result": {"category": 0, "duration": 6000}}
        result['result']['category'] = category
        # result['result']['category'] = 1
        result['result']['duration'] = int(np.round((duration) * 1000))
        return result


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        # super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.capture = 'test.mp4'
        self.model_name = model_name
        self.model_path = model_path
        self.model = yolo_model(source = self.capture)

    def _inference(self, data):
        result = self.model.inference()
        # result = nanodet_run(source = self.capture)
        return result

    def _preprocess(self, data):
        # 这个函数把data写到test.mp4里面了
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _postprocess(self, data):
        return data
