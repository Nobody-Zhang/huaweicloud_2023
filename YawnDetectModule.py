from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import imutils
import time
import dlib
import cv2

class YawnDetection:
    def __init__(self,detector_path='mmod_human_face_detector.dat',
                 predictor_path='./shape_predictor_68_face_landmarks.dat'):
        self.detector = dlib.cnn_face_detection_model_v1(detector_path)
        self.predictor = dlib.shape_predictor(predictor_path)
        self.COUNTER = 0
        self.TOTAL = 0
        self.mCOUNTER = 0
        self.mTOTAL = 0

        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.landmark_parameters_dict = {
            'EYE_AR_THRESH': 0.2,  # 眼睛长宽比
            'EYE_AR_CONSEC_FRAMES': 3,  # 闪烁阈值
            'MAR_THRESH': 0.5,  # 打哈欠长宽比
            'MOUTH_AR_CONSEC_FRAMES': 3,  # 闪烁阈值
        }

    @staticmethod
    def eye_aspect_ratio(eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear
    @staticmethod
    def mouth_aspect_ratio(mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def detect(self,img)->int:
        dets = self.detector(img,1)
        print(type(dets))
        for i,d in enumerate(dets):
            shape = self.predictor(img,d.rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = self.mouth_aspect_ratio(mouth)

            if mar > self.landmark_parameters_dict['MAR_THRESH']:
                self.mCOUNTER += 1
                #*-*-*-*-*-*-*-*-*-
                #在只接受一张图片的情况下，如果判断为打哈欠，就直接返回了
                return 2
                #*-*-*-*-*-*-*-*-*-*-*
            '''
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if self.mCOUNTER >= self.landmark_parameters_dict['MOUTH_AR_CONSEC_FRAMES']:
                    self.mTOTAL += 1
                # 重置嘴帧计数器
                self.mCOUNTER = 0
            '''


            if ear < self.landmark_parameters_dict['EYE_AR_THRESH']:
                self.COUNTER += 1
                #-*-*-*-*-*-*-*-*-*-*-*-*--
                # 在只接受一张图片的情况下，直接返回，认为闭眼了
                return 1
                #*-*-*-*-*-*-*-*-*-*-*-*-*-*-
            '''
            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if self.COUNTER >= self.landmark_parameters_dict['EYE_AR_CONSEC_FRAMES']:
                    self.TOTAL += 1
                # 重置眼帧计数器
                self.COUNTER = 0
            '''
            return 0

'''
        if self.TOTAL >= 10 and self.mTOTAL <= 3:
            # print('closing eyes, but not yawning')
            return 1
        elif self.TOTAL >= 10 and self.mTOTAL >= 3:
            # print('yawning and closing eyes')
            return 2
        elif self.TOTAL <= 10 and self.mTOTAL >= 3:
            print('just yawning but not closing eyes')
        self.TOTAL = 0
        self.mTOTAL = 0
'''



if __name__ == '__main__':
    img = cv2.imread('./pics/9.jpg')
    # img = dlib.load_rgb_image('./pics/9.jpg')

    data_path='./shape_predictor_68_face_landmarks.dat'
    yawn_detection = YawnDetection()
    result = yawn_detection.detect(img)
    print(result)
