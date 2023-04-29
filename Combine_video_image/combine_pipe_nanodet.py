from multiprocessing import Process,Manager,Event
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
import numpy as np


# import video classfication
from NanodetOpenvino.Nanodet import NanoDet
from MT_helpers.My_Transformer import *
# import SVM-image classfication
import svm.svmdetect as svmdetect


# 抽帧
FRAME_GROUP = 10
# 设置五种状态的编号
NORMAL = 0
EYE_CLOSE = 1
YAWN = 2
PHONE = 3
SIDE_FACE = 4

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--video_model", default = "nanodet", help="video classfication model name")
     parser.add_argument("--image_model", default="svm",help="image classfication model name")
     parser.add_argument("--mouth_model", default="./svm/svm_model_mouth.pkl",help="yawn classfication model name")
     parser.add_argument("--eye_model", default="./svm/svm_model_eyes.pkl",help="eye classfication model name")
     parser.add_argument("--trans_model", default="./MT_helpers/transformer_ag_model.pth", help="transformer model")
     parser.add_argument("--path", default="./day_man_001_10_1.mp4", help="path to video")
     parser.add_argument("--device", default="cpu", help="device for model use")

     return parser.parse_args()

# SVM 线程类
class SVM_Eye_Process(Process):
     def __init__(self,eye_status_list,queue,stop_event,model_path='./svm/svm_model_eyes.pkl'):
          self.classifier = svmdetect.ImageClassifier(model_path)
          self.status_list = eye_status_list
          self.queue = queue
          self.stop_event =stop_event
     
     # 灰度处理、resize处理 for SVM
     def img_transform(self,eye_img):
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
                    print((t2-t1))
                    print("Eye_svm推理结果:")
                    print(y_pred[0])

# SVM 线程类
class SVM_Mouth_Process(Process):
     def __init__(self,yawn_status_list,queue,stop_event,model_path='./svm/svm_model_mouth.pkl'):
          self.classifier = svmdetect.ImageClassifier(model_path)
          self.status_list = yawn_status_list
          self.queue = queue
          self.stop_event = stop_event
     
     # 灰度处理、resize处理 for SVM
     def img_transform(self,mouth_img):
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
                    print((t2-t1))
                    print("Mouth_svm推理结果:")
                    print(y_pred[0])



# 组合Nanodet/Yolo + SVM/MobileNetV2
class Combination:
     def __init__(self,eye_status_list,yawn_status_list,eye_queue,yawn_queue,eyestop_event,yawnstop_event,
                  mouth_model = "./svm/svm_model_mouth.pkl",
                  eye_model = "./svm/svm_model_eyes.pkl",
                  device = torch.device("cpu")):

          # 选择video classification model
          self.device = torch.device(device)
          self.video_model = NanodetOpenvino_init()

          # 选择image classification model,并创建两个线程类
          if image_model_name == "svm":
              self.image_eye = SVM_Eye_Process(eye_status_list,eye_queue,eyestop_event,eye_model)
              self.image_mouth = SVM_Mouth_Process(yawn_status_list,yawn_queue,yawnstop_event,mouth_model)
          elif image_model_name == "mobilenet":
              self.image_mouth = MobileNet_Yawn_Process(yawn_status_list,yawn_queue,yawnstop_event,mouth_model)
              self.image_eye = MobileNet_Eye_Process(eye_status_list,eye_queue,eyestop_event,eye_model)

     def NanodetOpenvino_init(self,nanodet_model1 = "./NanodetOpenvino/convert_for_two_stage/seg_face/seg_face.xml",
               nanodet_model2 = "./NanodetOpenvino/convert_for_two_stage/face_eyes/mouth_eyes.xml",
               num_class1 = 4, num_class2 = 2):

          # get 2 model
          nanodet_face = NanoDet(nanodet_model1 , num_class1)
          nanodet_eye_mouth = NanoDet(nanodet_model2 , num_class2)

          return [nanodet_face,nanodet_eye_mouth]

     def Nanodet_init(self, nanodet_cfg1="./Nanodet/demo/nanodet-plus-m_416-yolo.yml",
                      nanodet_model1="./Nanodet/demo/model_last.ckpt",
                      nanodet_cfg2="./Nanodet/demo/face_eyes.yml", nanodet_model2="./Nanodet/demo/face_eyes.ckpt"):

          local_rank = 0
          torch.backends.cudnn.enabled = True
          torch.backends.cudnn.benchmark = True
          load_config(cfg1, nanodet_cfg1)
          load_config(cfg2, nanodet_cfg2)
          logger = Logger(local_rank, use_tensorboard=False)
          # get 2 model
          nanodet_face = Predictor(cfg1, nanodet_model1, logger, self.device)
          nanodet_eye_mouth = Predictor(cfg2, nanodet_model2, logger, self.device)

          return [nanodet_face, nanodet_eye_mouth]

     def Yolo_init(self):
          """need to do Yolo classification"""

# 根据output的状态决定该图片是哪一种状态           
def SVM_Determin(eye_status,yawn_status,output,transform_path):
     for i in range(len(eye_status)):
          # 首先判断是否打哈欠了
          if yawn_status[i] == -1:
               output.append(YAWN)
          # 如果没有打哈欠 但是闭眼
          elif eye_status[i] == -1:
               output.append(EYE_CLOSE)
          else:
               output.append(NORMAL)

     result = Transform_result(transform_path,output)
     print(result[0])


# 根据output的状态决定该图片是哪一种状态           
def Mobilenet_Determin(eye_status,yawn_status,output):
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

def Transform_result(model_path,status_list):
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
     transform_path = args.trans_model
     device = args.device
     if args.device == 'gpu':
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     else:
          device = torch.device("cpu")

     mouth_model_path = args.mouth_model
     eye_model_path = args.eye_model

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

     Combine_model = Combination(eye_status_list,yawn_status_list,eye_queue,yawn_queue,eyestop_event,yawnstop_event,mouth_model_path, eye_model_path, device)
     
     cap = cv2.VideoCapture(video_path)

     thread_t1 = time.time()
     # 设定线程函数
     eye_process = Process(target = Combine_model.image_eye.inference)
     yawn_process = Process(target = Combine_model.image_mouth.inference)

     # 设置守护进程
     eye_process.daemon = True
     yawn_process.daemon = True

     all_start = time.time()
     cnt = 0

     while True:
          ret_val, frame = cap.read()
          if not ret_val:
               break

          nanodet_t1 = time.time()
          # # 抽帧：取每组的第一帧
          cnt += 1
          if cnt % FRAME_GROUP != 1:
               continue

          # 识别人脸
          # face_boxs = Combine_model.video_model[0].detect(frame,0.8, 0.8)
          # face_img = face_boxs[0]

          # 识别人脸
          _,face_img = Combine_model.video_model[0].find_face(frame)

          # 判断能否裁出人脸
          if face_img is None:
               continue

          # 获得眼部和嘴部图片
          eye0_img,eye1_img,mouth_img = Combine_model.video_model[1].find_eye_mouth(face_img)

          # 获得眼部和嘴部图片
          # mouth_eye_boxs = Combine_model.video_model[1].detect(frame,0.6, 0.8)
          # wait to process
          # eye_img = frame
          # mouth_img = frame
          # if cnt%5 == 1:
          #   cv2.imwrite(f"eye{cnt}.png",eye_img)

          nanodet_t2 = time.time()
          print("nanodet时间")
          print(nanodet_t2 - nanodet_t1)
          eye_queue.put(eye0_img)
          yawn_queue.put(mouth_img)

     # 结束线程
     # 启动线程
     eye_process.start()
     yawn_process.start()

     eyestop_event.set()
     yawnstop_event.set()
     eye_process.join()
     yawn_process.join()
     print("End")
     all_end = time.time()
     print((all_end - all_start)/cnt)

     # 状态判断
     # Mobilenet_Determin(eye_status_list,yawn_status_list,output)
     SVM_Determin(eye_status_list,yawn_status_list,output,transform_path)
     print(eye_status_list)
     print(yawn_status_list)



