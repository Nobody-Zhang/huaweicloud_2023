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
        self.cls_ = {"close_eye": 0, "close_mouth": 1, "face": 2, "open_eye": 3, "open_mouth": 4, "phone": 5,
                     "sideface": 6}
        self.status_prior = {"normal": 0, "closeeye": 1, "yawn": 3, "calling": 4, "turning": 2}
        self.condition = [0, 1, 4, 2, 3]

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

    #Yolo style nanodet determin function
    def determin(self, img: np.ndarray) -> int:
        """
        determin which status this frame belongs to\n
        0 -> normal status\n
        1 -> close eye\n
        2 -> yawn\n
        3 -> calling\n
        4 -> turning around\n

        :param img: input image, format the same as detect function
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
        bboxes = self.model(img)[0]
        for box in bboxes:  # 遍历每个box
            xyxy = (box.xmin, box.ymin, box.xmax, box.ymax)  # xyxy坐标
            xywh = xyxy2xywh(*xyxy, wide, height)  # xywh坐标
            conf = box.score  # 可信度
            cls = box.id  # 类别
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
    