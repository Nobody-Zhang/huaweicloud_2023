import os
import numpy as np

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
    return (x, y, w, h)

# Exyended Nanodet class from model_zoo
class NanoDet():

    # Nanodet class
    def __init__(self, model_path: str, num_class: int):
        """
        init Nanodet model
        
        :param model_path: path for model *.xml
        :param num_class: number of classes
        :return: None
        """
        model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")
        self.model = nanodet_model = NanoDetPlus(model_adapter, configuration={'num_classes': num_class}, preload=True)

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
        wide, height = img.shape[1], img.shape[0]
        driver = (0, 0, 0, 0)
        driver_xyxy = (0, 0, 0, 0)
        phone = (0, 0, 0, 0)
        sideface = (0, 0, 0, 0)
        bboxes = self.model(img)[0]
        # find driver, sideface and phone in bboxes
        for box in bboxes:
            xyxy = (box.xmin, box.ymin, box.xmax, box.ymax)
            xywh = xyxy2xywh(*xyxy, wide, height)
            cls = box.id
            if cls == 0:
                if .4 < xywh[0] and .2 < xywh[1] and xywh[1] > driver[1]:
                    if xywh[0] > driver[0]:
                        driver = xywh
                        driver_xyxy = xyxy
            elif cls == 2:
                if .4 < xywh[0] and .2 < xywh[1] and xywh[1] > phone[1]:
                    if xywh[0] > phone[0]:
                        phone = xywh
            elif cls == 1 or cls == 3:
                if .4 < xywh[0] and .2 < xywh[1] and xywh[1] > sideface[1]:
                    if xywh[0] > sideface[0]:
                        sideface = xywh
        # judge the driver status
        if driver[0] > sideface[0] and 0 < abs(driver[0] - phone[0]) < .2:
            return (1, None)
        elif sideface[0] > driver[0]:
            return (0, None)
        elif driver_xyxy[0] != 0:
            face_img = img[driver_xyxy[1]:driver_xyxy[3], driver_xyxy[0]:driver_xyxy[2]]
            return (2, face_img)
        else:
            return (-1, None)

    # find eyes and mouth in the face
    def find_eye_mouth(self, img: np.ndarray) -> tuple:
        """
        finde eyes and mouth in face_img\n
        if any of the returns is None, nothing is detected

        :param img: face image
        :returns: a tuple of eye1, eye2, and mouth, format np.ndarray
        """
        bboxes = self.model(img)[0]
        eyes = None
        mouth = None
        for box in bboxes:
            xyxy = (box.xmin, box.ymin, box.xmax, box.ymax)
            cls = box.id
            if cls == 1:
                mouth = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            elif cls == 0:
                eye1 = img[xyxy[1]:xyxy[3], xyxy[0]:(xyxy[0]+xyxy[2])//2]
                eye2 = img[xyxy[1]:xyxy[3], (xyxy[0]+xyxy[2])//2:xyxy[2]]
                eyes = (eye1, eye2)
        return (eyes[0], eyes[1], mouth)
