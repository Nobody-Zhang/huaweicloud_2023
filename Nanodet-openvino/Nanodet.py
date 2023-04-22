import os
import time
import sys
from typing import Tuple
import math

import cv2
import numpy as np
from openvino.runtime import Core

sys.path.append("../utils")

# generate grid
def generate_grid_center_priors(input_height: int, input_width: int, strides: tuple, center_priors: list) -> None:
    """
    generate
    """
    for i in range(len(strides)):
        stride = strides[i]
        feat_w = math.ceil(input_width / stride)
        feat_h = math.ceil(input_height / stride)
        for y in range(feat_h):
            for x in range(feat_w):
                center_priors.append(CenterPrior(x, y, stride))
                
# softmax function for decode
def activation_functionn_softmax(src: list, dst: list, length: int):        
    """
    :param src: source data
    :param dst: result data
    :param length: data length
    :return: not uesd
    """
    alpha = max(src)
    denominator = 0
    
    for i in range(length):
        dst[i] = math.exp(src[i] - alpha)
        denominator += dst[i]
    
    for i in range(length):
        dst[i] /= denominator
    
    return 0
    
# CenterPrior structure
class CenterPrior:
    
    # create CenterPrior
    def __init__(self, x: int, y: int, stride: int):
        """
        :param x: x
        :param y: y
        :param stride: stride
        """
        self.x = x
        self.y = y
        self.stride = stride
                
# BoxInfo structure
class BoxInfo:
    
    # create BoxInfo
    def __init__(self, x1: float, y1: float, x2: float, y2: float, score: float, label: int):
        """
        This class is used to store detectionn result
        
        :param x1: x_min
        :param y1: y_min
        :param x2: x_max
        :param y2: y_max
        :param score: confident score
        :param label: detected label result
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.label = label
     
    # string function, for debug 
    def __str__(self):
        """
        Nothing, Don't care
        """
        return f'BoxInfo:\nx1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}, score: {self.score}, label: {self.label}'
    
# NanoDet Class
class NanoDet:
    
    # init function
    def __init__(self, model_path: str, num_class: int):
        """
        NanoDet model for detection
        
        :param model_path: string path of the model
        :param num_class: number of classes
        :return: self
        """
        # Initialize inference engine runtime
        self.ie_core = Core()
        # Initialize Model
        self.input_keys, \
        self.output_keys, \
        self.compiled_model = \
            self.model_init(model_path)
        self.height, self.width = list(self.input_keys.shape)[2:]
        # private params
        self.input_size = (416, 416)
        self.num_class = num_class
        self.reg_max = 7
        self.strides = (8, 16, 32, 64)
        
    # Init Network and Weights 
    def model_init(self, model_path: str) -> Tuple:
        """
        Read the Network and Weights, load the model on CPU
        
        :param model_path: model's path for *.xml
        :returns:
                input_key: Input Node Network
                output_key: Output Node Network
                compiled_model: Encoder Model Network
        """
        
        # Read the Network and Weights
        model = self.ie_core.read_model(model=model_path)
        # Compile model for CPU
        compiled_model = self.ie_core.compile_model(model=model, device_name="CPU")
        # Get Input and OUTPUT imformation
        input_keys = compiled_model.input(0)
        output_keys = compiled_model.output(0)
        
        return input_keys, output_keys, compiled_model
    
    # preprocess the input Image
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        preprocess the Input Image
        
        :param img: input Image, numpy.ndarray type, just the cv2.imread result
        :return: processed Image, numpy.ndarray type
        """
        # resize image
        resized_image = cv2.resize(img, (self.width, self.height))
        # Expand dims
        input_img = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        
        return input_img
    
    # detect function
    def detect(self, img: np.ndarray, score_threshold: float, nms_threshold: float) -> list:
        """
        Object Detection Inference
        
        :param img: Input Image
        :return: detect result boxes list
        """
        start_time = time.time()
        # preprocess
        img = self.preprocess(img)
        # Run Inference and process the result
        pred = self.compiled_model([img])[self.output_keys]
        print(pred.shape)
        pred = np.squeeze(pred, 0)
        pred = pred.flatten()
        # generate center priors in format of (x, y, stride)
        center_priors = []
        generate_grid_center_priors(self.input_size[0], self.input_size[1], self.strides, center_priors)
        # decode outputs
        results = self.decode_infer(pred, center_priors, score_threshold)
        # NMS
        dets = []
        for i in range(len(results)):
            results[i] = self.nms(results[i], nms_threshold)
            for box in results[i]:
                dets.append(box)
        print(f'time cost: {1000*(time.time()-start_time)}ms')
        return dets

    
    # decode the infer result
    def decode_infer(self, pred, center_priors: list, threshold: float) -> list:
        """
        decode the infer result\n
        You don't need to Know what is this
        
        :param pred: output of the network
        :param center_priors: center priors
        :param threshold: score threshold
        :return: decoded result
        """
        num_points = len(center_priors)
        num_channels = self.num_class + (self.reg_max + 1) * 4
        results = [[] for i in range(self.num_class)]
        
        for idx in range(num_points):
            ct_x = center_priors[idx].x
            ct_y = center_priors[idx].y
            stride = center_priors[idx].stride
            
            score = 0.
            cur_label = 0
            
            for label in range(self.num_class):
                if pred[idx * num_channels + label] > score:
                    score = pred[idx * num_channels + label]
                    cur_label = label
            if score > threshold:
                bbox_pred = pred[idx * num_channels + self.num_class:]
                results[cur_label].append(self.disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride))
    
        return results
    
    # disPred2Bbox function, generate pred imformation to BoxInfo
    def disPred2Bbox(self, dfl_det: list, label: int, score: float, x: int, y: int, stride: int) -> BoxInfo:
        """
        You don't need to Know What is this
        
        :param dfl_det: datas
        :param label: label
        :param score: score
        :param x: x
        :param y: y
        :param stride: stride
        :return: generated BoxInfo
        """
        ct_x = x * stride
        ct_y = y * stride
        # generate dis pred
        dis_pred = [0. for k in range(4)]
        for i in range(4):
            dis = 0.
            dis_after_sm = [0. for k in range(self.reg_max + 1)]
            activation_functionn_softmax(dfl_det[i+(self.reg_max + 1):], dis_after_sm, self.reg_max + 1)
            for j in range(self.reg_max + 1):
                dis += j * dis_after_sm[j]
            dis *= stride
            dis_pred[i] = dis
        
        # calculate Bbox
        xmin = max(ct_x - dis_pred[0], 0.)
        ymin = max(ct_y - dis_pred[1], 0.)
        xmax = min(ct_x + dis_pred[2], float(self.input_size[1]))
        ymax = min(ct_y + dis_pred[3], float(self.input_size[0]))
        
        return BoxInfo(xmin, ymin, xmax, ymax, score, label)
        
    # nms
    def nms(self, input_boxes: list, NMS_THRESH: float) -> list:
        """
        nms function\n
        Processed data will be changed in input_boxes\n
        You don't need to Know What is this
        
        :param input_boxes: input data
        :param NMS_THRESH: Threshold
        :return: None
        """
        input_boxes.sort(key=lambda x: x.score)
        vArea = [0. for k in range(len(input_boxes))]
        for i in range(len(input_boxes)):
            vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1)*(input_boxes[i].y2 - input_boxes[i].y1 + 1)
        check_nms = [True for k in range(len(input_boxes))]
        for i in range(len(input_boxes)):
            for j in range(i+1, len(input_boxes)):
                xx1 = max(input_boxes[i].x1, input_boxes[j].x1)
                yy1 = max(input_boxes[i].y1, input_boxes[j].y1)
                xx2 = max(input_boxes[i].x2, input_boxes[j].x2)
                yy2 = max(input_boxes[i].y2, input_boxes[j].y2)
                w = max(0., xx2 - xx1 + 1)
                h = max(0., yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (vArea[i] + vArea[j] - inter)
                if ovr >= NMS_THRESH:
                    check_nms[j] = False
                else:
                    j += 1
        result_boxes = []
        for i in range(len(input_boxes)):
            if check_nms[i]:
                result_boxes.append(input_boxes[i])
                
        return result_boxes