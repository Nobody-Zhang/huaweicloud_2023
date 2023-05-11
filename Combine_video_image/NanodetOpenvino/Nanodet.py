import os
import cv2
import numpy as np
import time

# import model wrapper class
from openvino.model_zoo.model_api.models import NanoDetPlus
# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core


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
    x = ((xmin+xmax)//2)/wide
    y = ((ymin+ymax)//2)/height
    w = (xmax-xmin)/wide
    h = (ymax-ymin)/height
    return x, y, w, h


# Exyended Nanodet class from model_zoo
class NanoDet:

    # Nanodet class
    def __init__(self, model_path: str, num_class: int, threshold: float):
        """
        init Nanodet model
        
        :param model_path: path for model *.xml
        :param num_class: number of classes
        :return: None
        """
        self.cnt = 0
        model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")
        self.model = nanodet_model = NanoDetPlus(model_adapter, configuration={'num_classes': num_class, 'iou_threshold': threshold}, preload=True)

    # detetcion inference
    def detect(self, img: np.ndarray) -> list:
        """
        detect inference

        :param img: input image, format np.ndarray
        :return: list of bbox, on type of Detecion class
        """
        return self.model(img)

    # find driver's face
    def find_face(self, img: np.ndarray) -> tuple:
        """
        entened process to find main driver's face in the detected bboxes\n
        for the return tuple:\n
        if status is -1, wrong status, no face of sideface is detected, img will be None;\n
        if status is 0, the driver is turning head and no face is cutted, img will be None;\n
        if status is 1, the driver is using phone and no face is cutted, img will be None;\n
        if status is 2, the face is cutted and returned in img, on type of np.ndarray.\n

        :param img: input image, format the same as detect function
        :returns: a tuple, like (status, img)
        """
        # init something
        wide, height = img.shape[1], img.shape[0] # 输入图片宽、高
        driver = (0, 0, 0, 0) # 司机正脸xyxy坐标
        driver_xyxy = (0, 0, 0, 0) # 司机正脸xywh坐标
        phone = (0, 0, 0, 0) # 手机xyxy坐标
        sideface = (0, 0, 0, 0) # 司机侧脸xywh坐标
        
        nat = time.time()
        bboxes = self.model(img)[0]
        nat = time.time() - nat
        # print("nanodet infer time")
        # print(nat)
        # find driver, sideface and phone in bboxes
        for box in bboxes: # 遍历每个box
            xyxy = (box.xmin, box.ymin, box.xmax, box.ymax)
            xywh = xyxy2xywh(*xyxy, wide, height)
            cls = box.id
            if cls == 0: # 标签0，正脸
                if .5 < xywh[0] and xywh[1] > driver[1] and xyxy[3]/height > .25: # box中心在右侧0.5 && box中心在司机下方 && 右下角y轴在0.25以下
                    if xywh[0] > driver[0]: # box中心在司机右侧，即找最右下角的正脸
                        driver = xywh # 替换司机
                        driver_xyxy = xyxy # 替换司机
            elif cls == 1: # 标签1，手机
                if .4 < xywh[0] and .2 < xywh[1] and xywh[1] > phone[1]: # box中心在右侧0.4 && box中心在下侧0.2 && box中心在手机下方
                    if xywh[0] > phone[0]: # box中心在手机右侧，即找最右下角的手机
                        phone = xywh # 替换手机
            elif cls == 3 or cls == 2: # 标签3，2 头顶和侧脸，都视为侧脸
                if .5 < xywh[0] and xywh[1] > sideface[1] and xyxy[3]/height > .25: # box中心在右侧0.5 && box中心在侧脸下方 && 右下角y轴在0.25以下
                    if xywh[0] > sideface[0]: # box中心在侧脸右侧，即找最右下角的侧脸
                        sideface = xywh  # 替换侧脸
        # judge the driver status
        if driver[0] > sideface[0] and 0 < abs(driver[0] - phone[0]) < .3 and 0 < abs(driver[1] - phone[1]) < .3: # 最右边是正脸，正脸与手机相对位置xy不大于0.3
            return 1, None # 一眼定帧，鉴定为打电话
        elif driver[0] < sideface[0] and 0 < abs(sideface[0] - phone[0]) < .3 and 0 < abs(sideface[1] - phone[1]) < .3: # 最右边是侧脸，侧脸与手机相对位置xy不大于0.3
            return 1, None # 一眼定帧，鉴定为打电话
        elif sideface[0] > driver[0] or (abs(driver[0] - sideface[0]) < .1 and abs(driver[1] - sideface[1]) < .1): # 侧脸在正脸右边或者侧脸与正脸很近（相对位置xy不大于0.1，说明一个头既是正脸又是侧脸）
            return 0, None # 一眼定帧，鉴定为左顾右盼
        elif driver_xyxy[0] != 0: # 都不是，而且有正脸
            face_img = img[driver_xyxy[1]:driver_xyxy[3], driver_xyxy[0]:driver_xyxy[2]] # 把正脸切下来
            return 2, face_img # 一眼定帧，鉴定为鉴定不了，送stage2
        else: # 啥玩儿都不是
            return -1, None # 一眼定帧，拒绝鉴定

    # find eyes and mouth in the face
    def find_eye_mouth(self, img: np.ndarray, ) -> tuple:
        """
        finde eyes and mouth in face_img\n
        if any of the returns is None, nothing is detected\n
        if flag1 is False, mouth is not detected\n
        if flag2 is False, eye is not detected

        :param img: face image
        :returns: a tuple of (flag1, mouth, flag2, eye), format np.ndarray
        """
        bboxes = self.model(img)[0]
        eye = None # 眼睛图片
        eye_xywh = [0, 0, 0, 0] # 眼睛xywh坐标
        mouth = None # 嘴图片
        mouth_xywh = [0, 0, 0, 0] # 嘴xywh坐标
        flag1 = False # 是否识别出嘴
        flag2 = False # 是否识别出眼睛
        for box in bboxes: # 遍历每个box
            xyxy = (box.xmin, box.ymin, box.xmax, box.ymax)
            xywh = xyxy2xywh(*xyxy, img.shape[1], img.shape[0])
            cls = box.id
            if cls == 1 and xywh[1] > mouth_xywh[1]: # 标签1，嘴，并且box在嘴的下方
                flag1 = True # 找到新的嘴了
                mouth_xywh = xywh # 替换嘴
                mouth = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] # 把嘴切下来
            elif cls == 0 and xywh[1] > eye_xywh[1]: # 标签0，眼睛，并且box在眼睛的下方
                flag2 = True # 找到新的眼睛了
                eye_xywh = xywh # 替换眼睛
                eye = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] # 把眼睛切下来
        
        return flag1, mouth, flag2, eye
