import os
import sys
import json
import time
from torch import nn
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
import numpy as np

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg1, cfg2, load_config, load_model_weight
from nanodet.util.path import mkdir

from nanodet_twostages import Predictor
from Mobile_inference import Inference


def main(nanodet_cfg1 = "../config/nanodet-plus-m_416-yolo.yml",nanodet_model1 = "./model_last.ckpt",
         nanodet_cfg2 = "../config/face_eyes.yml",nanodet_model2 = "./face_eyes.ckpt",
         mobilenetv2_model = "./MobileNetV2_4class.pth",
         video_path = "../day_man_002_20_1.mp4",
         device=torch.device("cpu")):
        
        local_rank = 0
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        load_config(cfg1, nanodet_cfg1)
        load_config(cfg2, nanodet_cfg2)
        logger = Logger(local_rank, use_tensorboard=False)
        # get 2 model
        nanodet_face = Predictor(cfg1, nanodet_model1, logger, device)
        nanodet_eye_mouth = Predictor(cfg2, nanodet_model2, logger, device)
        
        mobilenetv2 = Inference(mobilenetv2_model , device)
        
        cap = cv2.VideoCapture(video_path)
        
        txt_out = open("./output/out.txt", 'w')
        start_time = time.time()
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta_face, res_face = nanodet_face.inference(frame)
                face_img = nanodet_face.finding_face(res_face,meta_face,0.5)
                
                # 先判断能否裁出人脸
                if face_img is None:
                    # print(4)
                    txt_out.write(str(4))
                    continue
                
                meta, res = nanodet_eye_mouth.inference(face_img)
                mouth_img = nanodet_eye_mouth.finding_mouth(res,meta,0.5)
                
                # 先判断是不是打哈欠 如果不是再判断闭眼
                inference_class = mobilenetv2.inference(mouth_img)
                if inference_class == 3:
                    # print(inference_class)
                    txt_out.write(str(inference_class))
                    continue
                
                # 判断眼睛是否闭着
                eyes_img = nanodet_eye_mouth.finding_eyes(res,meta,0.5)
                inference_class = mobilenetv2.inference(eyes_img)  
                # print(inference_class)
                txt_out.write(str(inference_class))
         
            else:
                break
            
        end_time = time.time()
        print("Time cost : {}".format(end_time - start_time))


if __name__ == '__main__':
    main()
        
        