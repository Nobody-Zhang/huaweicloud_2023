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

# from yolo.yolo import *

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


# 滑动窗口后处理，默认不抽帧，如果要抽帧就把所有的fps用fps/FRAME_GROUP代替
def Sliding_Window(tot_status, fps, thres1=2.45, thres2=0.48):
    window_status = {}  # 所有窗口的状态
    window_status_cnt = [0, 0, 0, 0, 0]
    single_window_cnt = [0, 0, 0, 0, 0]
    """
    window_status_cnt = {}  # 窗口状态计数
    window_status_cnt[0] = 0
    window_status_cnt[1] = 0
    window_status_cnt[2] = 0
    window_status_cnt[3] = 0
    window_status_cnt[4] = 0
    single_window_cnt = {}
    single_window_cnt[0] = 0
    single_window_cnt[1] = 0
    single_window_cnt[2] = 0
    single_window_cnt[3] = 0
    single_window_cnt[4] = 0
    """
    for i in range(len(tot_status) - int(2.5 * fps)):
        if i == 0:
            for j in range(int(2.5 * fps)):
                print(i + j)
                print(tot_status[i + j])
                print(type(tot_status[i + j]))
                single_window_cnt[int(tot_status[i + j])] += 1
        else:
            single_window_cnt[int(tot_status[i + int(2.5 * fps) - 1])] += 1
            single_window_cnt[int(tot_status[i - 1])] -= 1
        single_window_cnt[0] = -1  # 排除0
        max_cnt = 0

        # max_cnt = max(single_window_cnt, key=lambda x: single_window_cnt[x])
        for j in range(len(single_window_cnt)):
            if single_window_cnt[j] > single_window_cnt[max_cnt]:
                max_cnt = j
        if single_window_cnt[max_cnt] >= thres1 * fps:
            window_status[i] = max_cnt
        else:
            window_status[i] = 0
    for i in range(len(window_status)):
        window_status_cnt[int(window_status[i])] += 1
    print("window_status:", window_status)
    print("window_status_cnt:", window_status_cnt)
    window_status_cnt[0] = -1  # 排除0
    max_status = 0
    for i in range(len(window_status_cnt)):
        if(window_status_cnt[max_status] < window_status_cnt[i]):
            max_status = i
    # max_status = max(window_status_cnt, key=lambda x: window_status_cnt[x])
    if window_status_cnt[max_status] >= thres2 * fps:
        return max_status
    else:# 再来一次后处理
        cnt = [0, 0, 0, 0, 0]
        for i in range(len(tot_status)):
            cnt[tot_status[i]] += 1
        cnt[0] = -1
        ans = 0
        for i in range(5):
            if cnt[i] > cnt[ans]:
                ans = i
        if cnt[ans] >= 2.75 * fps:
            return ans
        return 0


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
    result = Sliding_Window(tot_status, fps)
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
        self.model = model()

    def _inference(self, data):
        cap = cv2.VideoCapture(self.capture)
        result = self.model.inference(cap)
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
