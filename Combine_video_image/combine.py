import threading
import time
import cv2
import numpy as np
import torch
from skimage.feature import hog
import argparse

# import video classfication
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg1, cfg2, load_config, load_model_weight
from nanodet.util.path import mkdir
from nanodet_twostages import Predictor

# import SVM-image classfication
import svmdetect

output = [0,0] # 分别表示eye detect和yawn detect的结果

def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--video_model", default = "nanodet", help="video classfication model name")
     parser.add_argument("--image_model", default="svm",help="image classfication model name")
     parser.add_argument("--path", default="./day_man_002_20_1.mp4", help="path to video")
     parser.add_argument("--device", default="cpu", help="device for model use")

     return parser.parse_args()


# SVM 线程类
class SVM_Clf_Thread:
     def __init__(self , model_path='./svm_model.pkl'):
          self.classifier = svmdetect.ImageClassifier(model_path)
     
     # Class 参数是区分eye（0）或yawn（1）,对于SVM而言 image 需要灰度 
     def inference(self,Class,image):
          output[Class] = 0  # 初始化为0
          features = self.classifier.extract_features(image)
          y_pred, time_cost = self.classifier.classify(features)
          output[Class] = y_pred
     
# Mobilenet 线程类
class MobileNet_Clf_Thread:
     def __init__(self,model_path = "./mobilenetv2_pth"):
          """need to complete"""


# 组合Nanodet/Yolo + SVM/MobileNetV2
class Combination:
     def __init__(self,video_model_name = "nanodet",image_model_name = "svm"
                  ,mouth_model = "./svm_model_mouth.pkl",eye_model = "./vm_model.pkl",device = "cpu"):
          # 选择video classification model
          self.device = torch.device(device);
          if video_model_name == "nanodet":
               self.video_model = self.Nanodet_init()
          elif video_model_name == "yolo":
               self.video_model = self.Yolo_init()
          
          # 选择image classification model,并创建两个线程类
          if image_model_name == "svm":
               self.image_mouth = SVM_Clf_Thread(mouth_model)
               self.image_eye = SVM_Clf_Thread(eye_model)
          elif image_model_name == "mobilenet":
               self.image_mouth = MobileNet_Clf_Thread(mouth_model)
               self.image_eye = MobileNet_Clf_Thread(eye_model)
                    
     def Nanodet_init(self,nanodet_cfg1 = "../config/nanodet-plus-m_416-yolo.yml",nanodet_model1 = "./model_last.ckpt",
          nanodet_cfg2 = "../config/face_eyes.yml",nanodet_model2 = "../face_eyes.ckpt"):
          
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
               
       

def main():
     args = parse_args()
     video_path = args.path
     video_model_name = args.video_model
     image_model_name = args.image_model
     device = args.device
     
     mouth_model_path = "./svm_model_mouth.pkl"
     eye_model_path = "./vm_model.pkl"
     
     Combine_model = Combination(video_model_name , image_model_name, mouth_model_path, eye_model_path, device)
     
     cap = cv2.VideoCapture(video_path)
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
          
          # 灰度处理、resize处理 for SVM
          gray_mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
          gray_mouth_img = cv2.resize(gray_mouth_img, (60, 36))
          gray_eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
          gray_eye_img = cv2.resize(gray_eye_img, (60, 36))
          
          # 设定线程函数
          eye_thread = threading.Thread(target = Combine_model.image_eye.inference,args = (0,gray_eye_img)) # 传入 0 表示是eye线程
          yawn_thread = threading.Thread(target = Combine_model.image_mouth.inference,args = (1,gray_mouth_img)) # 传入 1 表示是yawn线程
          
          # 启动线程
          eye_thread.start()
          yawn_thread.start() 

          # 对推理结果进行判断 需要等待
          eye_thread.join()
          yawn_thread.join()
          
          # 开始对两个判断进行比较
          # todo


