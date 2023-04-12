import threading
import time
from PIL import Image
import cv2
import numpy as np
from skimage.feature import hog
import argparse
import queue
from torch import nn
import torch
import json
from torchvision import transforms
from skimage.transform import resize


# import video classfication
from Combine_video_image.Nanodet.nanodet.util import Logger, cfg1, cfg2, load_config
from Combine_video_image.Nanodet.demo.nanodet_twostages import Predictor
from Combine_video_image.mobilenet.model_v2_gray import MobileNetV2

# import SVM-image classfication
import svm.svmdetect as svmdetect

# queue 用于存图片
eye_queue = queue.Queue()
yawn_queue = queue.Queue()
eye_status = []
yawn_status = []
output = [] # 分别表示eye detect和yawn detect的结果(1表示正常状态，-1表示不正常状态)

# 设置三种状态的编号
NORMAL = 0
EYE_CLOSE = 1
YAWN = 2

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--video_model", default = "nanodet", help="video classfication model name")
     parser.add_argument("--image_model", default="svm",help="image classfication model name")
     parser.add_argument("--mouth_model", default="./svm/svm_model_mouth.pkl",help="yawn classfication model name")
     parser.add_argument("--eye_model", default="./svm/svm_model.pkl",help="eye classfication model name")
     parser.add_argument("--path", default="./day_man_001_10_1.mp4", help="path to video")
     parser.add_argument("--device", default="cpu", help="device for model use")

     return parser.parse_args()


# SVM 线程类
class SVM_Eye_Thread(threading.Thread):
     # Class 参数是区分eye（0）或yawn（1）,对于SVM而言 image 需要灰度 
     def __init__(self,model_path='./svm/svm_model.pkl'):
          self.classifier = svmdetect.ImageClassifier(model_path)
          self.kill = 0
     
     # 灰度处理、resize处理 for SVM
     def img_transform(self,eye_img):
          gray_eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
          gray_eye_img = cv2.resize(gray_eye_img, (60, 36))
          return gray_eye_img
     
     def inference(self):
          print("Eye Thread start!\n")
          while True:
               if not eye_queue.empty():
                    image = eye_queue.get()
                    image = self.img_transform(image)
                    features = self.classifier.extract_features(image)
                    y_pred, time_cost = self.classifier.classify(features)
                    eye_status.append(y_pred)
                    print(eye_status)
               if self.kill == 1 and eye_queue.empty():
                    print("Eye Thread killed!\n")
                    break

# SVM 线程类
class SVM_Mouth_Thread(threading.Thread):
     # Class 参数是区分eye（0）或yawn（1）,对于SVM而言 image 需要灰度 
     def __init__(self,model_path='./svm/svm_model_mouth.pkl'):
          self.classifier = svmdetect.ImageClassifier(model_path)
          self.kill = 0
     
     # 灰度处理、resize处理 for SVM
     def img_transform(self,mouth_img):
          gray_mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
          gray_mouth_img = cv2.resize(gray_mouth_img, (60, 36))
          return gray_mouth_img
     
     def inference(self):
          print("Mouth Thread start!\n")
          while True:
               if not yawn_queue.empty():
                    image = yawn_queue.get()
                    image = self.img_transform(image)
                    features = self.classifier.extract_features(image)
                    y_pred, time_cost = self.classifier.classify(features)
                    yawn_status.append(y_pred)
                    # print(yawn_status)
               if self.kill == 1 and yawn_queue.empty():
                    print("Mouth Thread killed!\n")
                    break


# Mobilenet 线程类
class MobileNet_Yawn_Thread(threading.Thread):
     def __init__(self, weight_path="./mobilenet/MobileNetV2_yawnclass.pth",device=torch.device("cpu")):
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
          self.kill = 0
     
     def inference(self):
          print("Mouth Thread start!\n")
          while True:
               if not yawn_queue.empty():
                    # 处理图片
                    image = yawn_queue.get()
                    image = Image.fromarray(image)
                    image = self.transform(image)
                    image = torch.unsqueeze(image, dim=0)
                    
                    with torch.no_grad():
                         output = torch.squeeze(self.model(image.to(self.device))).cpu()
                         predict = torch.softmax(output, dim=0)
                         predict_cla = torch.argmax(predict).numpy()
                    
                    yawn_status.append(predict_cla)
                    # print(predict_cla)
               if self.kill == 1 and yawn_queue.empty():
                    print("Mouth Thread killed!\n")
                    break
          
class MobileNet_Eye_Thread(threading.Thread):
     def __init__(self, weight_path="./mobilenet/MobileNetV2_eyeclass.pth",device=torch.device("cpu")):
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
          self.kill = 0
     
     def inference(self):
          print("Eye Thread start!\n")
          while True:
               if not eye_queue.empty():
                    image = eye_queue.get()
                    image = Image.fromarray(image)
                    image = self.transform(image)
                    image = torch.unsqueeze(image, dim=0)
                    
                    with torch.no_grad():
                         output = torch.squeeze(self.model(image.to(self.device))).cpu()
                         predict = torch.softmax(output, dim=0)
                         predict_cla = torch.argmax(predict).numpy()
                    eye_status.append(predict_cla)
                    print(predict_cla)
               if self.kill == 1 and eye_queue.empty():
                    print("Eye Thread killed!\n")
                    break


# 组合Nanodet/Yolo + SVM/MobileNetV2
class Combination:
     # def __init__(self,video_model_name = "nanodet",image_model_name = "svm"
     #              ,mouth_model = "./svm/svm_model_mouth.pkl",eye_model = "./svm/svm_model.pkl",device = "cpu"):
     def __init__(self,video_model_name = "nanodet",image_model_name = "mobilenet"
                  ,mouth_model = "./mobilenet/MobileNetV2_mouthclass.pth",eye_model = "./mobilenet/MobileNetV2_eyeclass.pth"
                  ,device = "cpu"):     
          # 选择video classification model
          self.device = torch.device(device);
          if video_model_name == "nanodet":
               self.video_model = self.Nanodet_init()
          elif video_model_name == "yolo":
               self.video_model = self.Yolo_init()
          
          # 选择image classification model,并创建两个线程类
          if image_model_name == "svm":
               self.image_mouth = SVM_Eye_Thread(mouth_model)
               self.image_eye = SVM_Mouth_Thread(eye_model)
          elif image_model_name == "mobilenet":
               self.image_mouth = MobileNet_Yawn_Thread(mouth_model)
               self.image_eye = MobileNet_Eye_Thread(eye_model)

     def Nanodet_init(self,nanodet_cfg1 = "./Nanodet/demo/nanodet-plus-m_416-yolo.yml",
          nanodet_model1 = "./Nanodet/demo/model_last.ckpt",
          nanodet_cfg2 = "./Nanodet/demo/face_eyes.yml",nanodet_model2 = "./Nanodet/demo/face_eyes.ckpt"):
          
          local_rank = 0
          torch.backends.cudnn.enabled = True
          torch.backends.cudnn.benchmark = True
          load_config(cfg1, nanodet_cfg1)
          load_config(cfg2, nanodet_cfg2)
          logger = Logger(local_rank, use_tensorboard=False)
          # get 2 model
          nanodet_face = Predictor(cfg1, nanodet_model1, logger, self.device)
          nanodet_eye_mouth = Predictor(cfg2, nanodet_model2, logger, self.device)
          
          return [nanodet_face,nanodet_eye_mouth]
          
     def Yolo_init():
          """need to do Yolo classification"""

# 根据output的状态决定该图片是哪一种状态           
def SVM_Determin():
     for i in range(len(eye_status)):
          # 首先判断是否打哈欠了
          if yawn_status[i] == -1:
               output.append(YAWN)
          # 如果没有打哈欠 但是闭眼
          elif eye_status[i] == -1:
               output.append(EYE_CLOSE)
          else:
               output.append(NORMAL)
     print(output)

# 根据output的状态决定该图片是哪一种状态           
def Mobilenet_Determin():
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

def example():
     args = parse_args()
     video_path = args.path
     video_model_name = args.video_model
     image_model_name = args.image_model
     device = args.device
     if args.device == 'gpu':
          device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
     
     mouth_model_path = args.mouth_model
     eye_model_path = args.eye_model
     
     Combine_model = Combination(video_model_name , image_model_name, mouth_model_path, eye_model_path, device)
     
     cap = cv2.VideoCapture(video_path)
     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
     fps = cap.get(cv2.CAP_PROP_FPS)
     
     # 设定线程函数
     eye_thread = threading.Thread(target = Combine_model.image_eye.inference) 
     yawn_thread = threading.Thread(target = Combine_model.image_mouth.inference) 
     
     # 启动线程
     eye_thread.start()
     yawn_thread.start()
     
     while True:
          ret_val, frame = cap.read()
          if not ret_val:
               break
          
          # 识别人脸
          meta_face, res_face = Combine_model.video_model[0].inference(frame)
          face_img = Combine_model.video_model[0].finding_face(res_face,meta_face,0.5)

          # 判断能否裁出人脸
          if face_img is None:
               continue
                    
          # 获得眼部和嘴部图片
          meta, res = Combine_model.video_model[1].inference(face_img)
          mouth_img = Combine_model.video_model[1].finding_mouth(res,meta,0.5)
          eye_img = Combine_model.video_model[1].finding_eyes(res,meta,0.5)
          
          eye_queue.put(eye_img)
          yawn_queue.put(mouth_img)
     
     # 结束线程
     Combine_model.image_eye.kill = 1
     Combine_model.image_mouth.kill = 1
     # 开始对两个判断进行比较
     Mobilenet_Determin()

if __name__ == '__main__':
     example()