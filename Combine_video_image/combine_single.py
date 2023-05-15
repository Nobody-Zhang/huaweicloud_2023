from multiprocessing import Process, Manager, Event
import time
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import cv2
import argparse
import torch
import json
from torchvision import transforms
from torch import nn
from PIL import Image
from skimage.transform import resize
import numpy as np

# import video classfication
# from Nanodet.nanodet.util import Logger, cfg1, cfg2, load_config
# from Nanodet.demo.nanodet_twostages import Predictor
from mobilenet.model_v2_gray import MobileNetV2
from NanodetOpenvino.Nanodet import NanoDet
from MT_helpers.My_Transformer import *
# import SVM-image classfication
import svm.svmdetect as svmdetect

# fps = 30
# 抽帧
FRAME_GROUP = 6
# 设置三种状态的编号
NORMAL = 0
EYE_CLOSE = 1
YAWN = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model", default="nanodet", help="video classfication model name")
    parser.add_argument("--image_model", default="svm", help="image classfication model name")
    parser.add_argument("--mouth_model", default="./svm/svm_model_mouth.pkl", help="yawn classfication model name")
    parser.add_argument("--eye_model", default="./svm/svm_model_eyes.pkl", help="eye classfication model name")
    parser.add_argument("--trans_model", default="./MT_helpers/transformer_6fps_model.pth", help="transformer model")
    parser.add_argument("--path", default="./test_video/day_man_062_11_2.mp4",
                        help="path to video")
    parser.add_argument("--device", default="cpu", help="device for model use")

    return parser.parse_args()


# SVM 线程类
class SVM_Eye_Process(Process):
    def __init__(self, eye_status_list, queue, stop_event, model_path='./svm/svm_model_eyes.pkl'):
        self.classifier = svmdetect.ImageClassifier(model_path)
        self.status_list = eye_status_list
        self.queue = queue
        self.stop_event = stop_event

    # 灰度处理、resize处理 for SVM
    def img_transform(self, eye_img):
        gray_eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye_img = cv2.resize(gray_eye_img, (60, 36))
        return gray_eye_img

    def inference(self):
        print("Eye Thread start!\n")
        while not self.stop_event.is_set():
            if not self.queue.empty():
                t1 = time.time()
                image = self.queue.get()
                image = self.img_transform(image)
                image = [image]
                features = self.classifier.extract_features(image)
                y_pred, time_cost = self.classifier.classify(features)
                self.status_list.append(y_pred[0])
                t2 = time.time()
                print("Eye_svm时间:")
                print((t2 - t1))
                print("Eye_svm推理结果:")
                print(y_pred[0])


# SVM 线程类
class SVM_Mouth_Process(Process):
    def __init__(self, yawn_status_list, queue, stop_event, model_path='./svm/svm_model_mouth.pkl'):
        self.classifier = svmdetect.ImageClassifier(model_path)
        self.status_list = yawn_status_list
        self.queue = queue
        self.stop_event = stop_event

    # 灰度处理、resize处理 for SVM
    def img_transform(self, mouth_img):
        gray_mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
        gray_mouth_img = cv2.resize(gray_mouth_img, (100, 50))
        return gray_mouth_img

    def inference(self):
        print("Mouth Thread start!\n")
        while not self.stop_event.is_set():
            if not self.queue.empty():
                t1 = time.time()
                image = self.queue.get()
                image = self.img_transform(image)
                image = [image]
                features = self.classifier.extract_features(image)
                y_pred, time_cost = self.classifier.classify(features)
                self.status_list.append(y_pred[0])
                t2 = time.time()
                print("Mouth_svm时间:")
                print((t2 - t1))
                print("Mouth_svm推理结果:")
                print(y_pred[0])


def SVM_Handle(eye_queue, yawn_queue) -> tuple:
    eye_classifier = svmdetect.ImageClassifier('./svm/svm_model_eyes.pkl')
    mouth_classifier = svmdetect.ImageClassifier('./svm/svm_model_mouth.pkl')
    eye_gray = []
    yawn_gray = []
    """
    for i in yawn_queue:
        cv2.imshow('mouth', i)
        cv2.waitKey(50)
    """
    # 先进行灰度处理、resize处理
    for eye_img in eye_queue:
        # cv2.imshow("eye_img",eye_img)
        # cv2.waitKey(5000)
        gray_eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye_img = cv2.resize(gray_eye_img, (60, 36))
        eye_gray.append(gray_eye_img)
    for mouth_img in yawn_queue:
        gray_mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
        gray_mouth_img = cv2.resize(gray_mouth_img, (100, 50))
        yawn_gray.append(gray_mouth_img)
    """
    for i in eye_gray:
        cv2.imshow('eye', i)
        cv2.waitKey(50)
    """
    features_eyes = eye_classifier.extract_features(eye_gray)
    features_mouths = mouth_classifier.extract_features(yawn_gray)
    eye_pred, t1 = eye_classifier.classify(features_eyes)
    yawn_pred, t2 = mouth_classifier.classify(features_mouths)
    t = t1 + t2
    """
    for i in range(len(yawn_gray)):
        cv2.imshow(str(yawn_pred[i]), yawn_gray[i])
        cv2.waitKey(50)
    """

    for i in range(len(yawn_gray)):
        if eye_pred[i] == 1:
            continue
        cv2.imshow(str(eye_pred[i]), eye_gray[i])
        cv2.waitKey(50)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print(t)
    return eye_pred.tolist(), yawn_pred.tolist()
    # return Array.asList(eye_pred), yawn_pred.asList(num)


# Mobilenet 线程类
class MobileNet_Yawn_Process(Process):
    def __init__(self, yawn_status_list, queue, stop_event, weight_path="./mobilenet/MobileNetV2_yawnclass.pth",
                 device=torch.device("cpu")):
        # create model
        self.model = MobileNetV2(num_classes=2).to(device)
        # load model weights
        self.model_weight_path = weight_path
        self.device = device
        self.model.load_state_dict(torch.load(self.model_weight_path, map_location=device))
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #   transforms.Normalize([0.5], [0.5])
            transforms.Normalize([0.449], [0.226])
        ])
        # read class_indict
        self.json_path = "./mobilenet/yawnclass_indices.json"
        with open(self.json_path, "r") as f:
            self.class_indict = json.load(f)
        self.model = self.model.eval()

        self.status_list = yawn_status_list
        self.queue = queue
        self.stop_event = stop_event

    def inference(self):
        print("Mouth Thread start!\n")
        while not self.stop_event.is_set():
            if not self.queue.empty():
                mobile_t1 = time.time()
                # 处理图片
                image = self.queue.get()
                image = Image.fromarray(image)
                image = self.transform(image)
                image = torch.unsqueeze(image, dim=0)

                with torch.no_grad():
                    output = torch.squeeze(self.model(image.to(self.device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                mobile_t2 = time.time()
                self.status_list.append(int(predict_cla))
                print("mobile_yawn time:")
                print(mobile_t2 - mobile_t1)
                print(int(predict_cla))


class MobileNet_Eye_Process(Process):
    def __init__(self, eye_status_list, queue, stop_event, weight_path="./mobilenet/MobileNetV2_eyeclass.pth",
                 device=torch.device("cpu")):
        # create model
        self.model = MobileNetV2(num_classes=2).to(device)
        # load model weights
        self.model_weight_path = weight_path
        self.device = device
        self.model.load_state_dict(torch.load(self.model_weight_path, map_location=device))
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #   transforms.Normalize([0.5], [0.5])
            transforms.Normalize([0.449], [0.226])
        ])
        # read class_indict
        self.json_path = "./mobilenet/eyeclass_indices.json"
        with open(self.json_path, "r") as f:
            self.class_indict = json.load(f)
        self.model = self.model.eval()

        self.status_list = eye_status_list
        self.queue = queue
        self.stop_event = stop_event

    def inference(self):
        print("Eye Thread start!\n")
        while not self.stop_event.is_set():
            if not self.queue.empty():
                mobile_t1 = time.time()
                image = self.queue.get()
                image = Image.fromarray(image)
                image = self.transform(image)
                image = torch.unsqueeze(image, dim=0)

                with torch.no_grad():
                    output = torch.squeeze(self.model(image.to(self.device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                mobile_t2 = time.time()
                self.status_list.append(int(predict_cla))
                print("mobile_eye time:")
                print(mobile_t2 - mobile_t1)
                print(int(predict_cla))


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
        elif video_model_name == "yolo":
            self.video_model = self.Yolo_init()

        # 选择image classification model,并创建两个线程类（对CPU影响太大，故舍弃
        """
          if image_model_name == "svm":
              self.image_eye = SVM_Eye_Process(eye_status_list,eye_queue,eyestop_event,eye_model)
              self.image_mouth = SVM_Mouth_Process(yawn_status_list,yawn_queue,yawnstop_event,mouth_model)
          elif image_model_name == "mobilenet":
              self.image_mouth = MobileNet_Yawn_Process(yawn_status_list,yawn_queue,yawnstop_event,mouth_model)
              self.image_eye = MobileNet_Eye_Process(eye_status_list,eye_queue,eyestop_event,eye_model)
          """

    def Nanodet_init(self, nanodet_model1="./NanodetOpenvino/convert_for_two_stage/seg_face/seg_face.xml",
                     nanodet_model2="./NanodetOpenvino/convert_for_two_stage/face_eyes/nanodet.xml",
                     num_class1=4, num_class2=2, threshold1=0.8, threshold2=0.6):

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

    def Yolo_init(self):
        """need to do Yolo classification"""


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
    else:
        return 0



# 根据output的状态决定该图片是哪一种状态
def SVM_Determin(eye_status, yawn_status, transform_path, tot_status: list, fps):
    output = []
    for i in range(len(eye_status)):
        # 首先判断是否打哈欠了
        if yawn_status[i] == -1:
            output.append(YAWN)
        # 如果没有打哈欠 但是闭眼
        elif eye_status[i] == -1:
            output.append(EYE_CLOSE)
        else:
            output.append(NORMAL)
    j = 0
    for i in range(len(tot_status)):
        if tot_status[i] == -1:
            tot_status[i] = output[j]
            j = j + 1
    print(tot_status)
    # result = Transform_result(transform_path,output)
    result = Transform_result(transform_path, tot_status)
    # print(result[0])
    # result = Sliding_Window(tot_status, fps)
    print("result:", result)
    return tot_status


# 根据output的状态决定该图片是哪一种状态
def Mobilenet_Determin(eye_status, yawn_status, output):
    for i in range(len(eye_status)):
        # 首先判断是否打哈欠了
        if yawn_status[i] == 1:
            output.append(YAWN)
        # 如果没有打哈欠 但是闭眼
        elif eye_status[i] == 0:
            output.append(EYE_CLOSE)
        else:
            output.append(NORMAL)
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


if __name__ == '__main__':
    args = parse_args()
    video_path = args.path
    video_model_name = args.video_model
    image_model_name = args.image_model
    transform_path = args.trans_model
    device = args.device
    if args.device == 'gpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    mouth_model_path = args.mouth_model
    eye_model_path = args.eye_model

    """
     # 创建队列和Event
     eye_queue = Manager().Queue()
     yawn_queue = Manager().Queue()
     eyestop_event = Event()
     yawnstop_event = Event()
     eyestop_event.clear()
     yawnstop_event.clear()


     eye_status_list = Manager().list()
     yawn_status_list = Manager().list()
     output = []
     """

    Combine_model = Combination(video_model_name, image_model_name, mouth_model_path, eye_model_path, device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fps = fps / FRAME_GROUP
    thread_t1 = time.time()
    # 设定线程函数
    # eye_process = Process(target = Combine_model.image_eye.inference)
    # yawn_process = Process(target = Combine_model.image_mouth.inference)

    # 设置守护进程
    # eye_process.daemon = True
    # yawn_process.daemon = True

    # 启动线程
    # eye_process.start()
    # yawn_process.start()

    thread_t2 = time.time()
    print("线程启动时间")
    print(thread_t2 - thread_t1)

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

        nanodet_t1 = time.time()
        # # 抽帧：取每组的第一帧
        cnt += 1
        # if cnt % FRAME_GROUP != 1:
        #     continue

        # 识别人脸
        face_boxs = Combine_model.video_model[0].find_face(frame)
        face_img = face_boxs[1]
        # cv2.imshow('face',face_img)
        # cv2.waitKey(1000)
        # 判断能否裁出人脸
        if face_boxs[0] == -1 or face_boxs[0] == 0:
            tot_status.append(4)
        elif face_boxs[0] == 1:
            tot_status.append(3)
        else:
            mouth_eye_boxes = Combine_model.video_model[1].find_eye_mouth(face_img)
            if not mouth_eye_boxes[0] or not mouth_eye_boxes[2]:
                tot_status.append(4)
            else:
                eye_img = mouth_eye_boxes[3]
                mouth_img = mouth_eye_boxes[1]
                eye_queue.append(eye_img)
                yawn_queue.append(mouth_img)
                tot_status.append(-1)
                flag = True
                # 获得眼部和嘴部图片
                # cv2.imshow('eye',eye_img)
                # cv2.waitKey(1000)
                # cv2.imshow('mouth',mouth_img)
                # cv2.waitKey(1)
        nanodet_t2 = time.time()
        print("nanodet时间")
        print(nanodet_t2 - nanodet_t1)
        # eye_queue.append(eye_img)
        # yawn_queue.append(mouth_img)

    # 结束线程
    # eyestop_event.set()
    # yawnstop_event.set()
    # eye_process.join()
    # yawn_process.join()
    print("End")
    all_end = time.time()
    print((all_end - all_start) / cnt)

    print(tot_status)

    eye_status_list = []
    yawn_status_list = []
    if(flag):
        eye_status_list, yawn_status_list = SVM_Handle(eye_queue, yawn_queue)

    print(f'eye:{eye_status_list}')
    print(f'YAWN:{yawn_status_list}')
    # 状态判断
    # Mobilenet_Determin(eye_status_list,yawn_status_list,output)
    SVM_Determin(eye_status_list, yawn_status_list, transform_path, tot_status, fps)

"""
if __name__ == '__main__':
    vidio_dir = "./test_video/"
    save_dir = "./Transformer_data/"
    for fn in os.listdir(vidio_dir):
        fn = vidio_dir + fn
        tot_status = generate_data(fn)
        right = fn[-7]
        label = fn[-8]
        if right == 1:
            label = str(0)
        fp = open(f"./Transformer_data/{label}.txt", 'a')
        fp.write(tot_status + '\n')
        fp.close()
"""