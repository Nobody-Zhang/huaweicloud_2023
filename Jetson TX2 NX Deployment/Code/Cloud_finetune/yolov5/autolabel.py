import dlib
import cv2
import os
from pathlib import Path
import random
import shutil
import numpy as np

# 加载dlib模型和初始化人脸检测器
predictor_path = '/home/ma-user/modelarts/user-job-dir/yolov5/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# YOLO格式类别标签和索引
class_labels = ['closed_eye', 'open_eye', 'closed_mouth', 'open_mouth']
class_indices = {label: idx for idx, label in enumerate(class_labels)}
# 计算两点之间的距离
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 定义函数计算张闭程度
def compute_eye_mouth_openness(landmarks):
    # 根据关键点坐标计算眼睛和嘴巴的张闭程度
    left_eye_openness = distance(landmarks[37], landmarks[41]) + distance(landmarks[38], landmarks[40])
    right_eye_openness = distance(landmarks[43], landmarks[47]) + distance(landmarks[44], landmarks[46])
    mouth_openness = distance(landmarks[61], landmarks[67]) + distance(landmarks[62], landmarks[66]) + \
                     distance(landmarks[63], landmarks[65])
    return left_eye_openness, right_eye_openness, mouth_openness

# 处理单帧图像，生成Yolo标注数据
def process_frame(frame, annotation_file):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        left_eye_openness, right_eye_openness, mouth_openness = compute_eye_mouth_openness(landmarks_points)
        # 获取眼睛和嘴巴的bounding box坐标
        left_eye_bbox = (landmarks.part(36).x, landmarks.part(37).y, landmarks.part(39).x, landmarks.part(41).y)
        right_eye_bbox = (landmarks.part(42).x, landmarks.part(43).y, landmarks.part(45).x, landmarks.part(47).y)
        mouth_bbox = (landmarks.part(48).x, landmarks.part(51).y, landmarks.part(54).x, landmarks.part(57).y)

        # 将bounding box坐标转换为Yolo格式
        left_eye_bbox_yolo = convert_to_yolo_bbox(left_eye_bbox, frame.shape)
        right_eye_bbox_yolo = convert_to_yolo_bbox(right_eye_bbox, frame.shape)
        mouth_bbox_yolo = convert_to_yolo_bbox(mouth_bbox, frame.shape)

        # 保存Yolo标注数据
        if(left_eye_openness < 5):
            annotation = f"{class_indices['closed_eye']} {left_eye_bbox_yolo}"
        else:
            annotation = f"{class_indices['open_eye']} {left_eye_bbox_yolo}"
        annotation_file.write(annotation + '\n')

        if (right_eye_openness < 5):
            annotation = f"{class_indices['closed_eye']} {right_eye_bbox_yolo}"
        else:
            annotation = f"{class_indices['open_eye']} {right_eye_bbox_yolo}"
        annotation_file.write(annotation + '\n')

        if (mouth_openness < 5):
            annotation = f"{class_indices['closed_mouth']} {mouth_bbox_yolo}"
        else:
            annotation = f"{class_indices['open_mouth']} {mouth_bbox_yolo}"
        annotation_file.write(annotation + '\n')


# 将bounding box坐标转换为Yolo格式
def convert_to_yolo_bbox(bbox, frame_shape):
    x_center = (bbox[0] + bbox[2]) / (2 * frame_shape[1])
    y_center = (bbox[1] + bbox[3]) / (2 * frame_shape[0])
    width = (bbox[2] - bbox[0]) / frame_shape[1]
    height = (bbox[3] - bbox[1]) / frame_shape[0]
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# 主函数
if __name__ == "__main__":
    # 输入视频文件夹路径
    input_videos_path = '/home/ma-user/modelarts/inputs/data_url_0'

    # 输出文件夹路径
    output_folder = '/home/ma-user/modelarts/inputs/dataset/'
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path('/home/ma-user/modelarts/inputs/dataset/image').mkdir(parents=True, exist_ok=True)
    Path('/home/ma-user/modelarts/inputs/dataset/label').mkdir(parents=True, exist_ok=True)
    Path('/home/ma-user/modelarts/inputs/dataset/images/train').mkdir(parents=True, exist_ok=True)
    Path('/home/ma-user/modelarts/inputs/dataset/images/val').mkdir(parents=True, exist_ok=True)
    Path('/home/ma-user/modelarts/inputs/dataset/labels/train').mkdir(parents=True, exist_ok=True)
    Path('/home/ma-user/modelarts/inputs/dataset/labels/val').mkdir(parents=True, exist_ok=True)

    for video_name in os.listdir(input_videos_path):
        video_path = os.path.join(input_videos_path, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 创建标注文件
            annotation_folder = os.path.join(output_folder, 'label')
            images_folder = os.path.join(output_folder, 'image')
            annotation_file_path = os.path.join(annotation_folder, f'{video_name[:-4]}_{frame_idx}.txt')
            with open(annotation_file_path, 'w') as annotation_file:
                process_frame(frame, annotation_file)
            cv2.imwrite(os.path.join(images_folder, f'{video_name[:-4]}_{frame_idx}.jpg'), frame)
            frame_idx += 1
            print("Yolo annotations saved at:", annotation_file_path)
        cap.release()

    # 在linux系统下，要把“\\”改为“/”
    src_folder = "/home/ma-user/modelarts/inputs/dataset/image"
    dst_folder1 = "/home/ma-user/modelarts/inputs/dataset/images/train"
    dst_folder2 = "/home/ma-user/modelarts/inputs/dataset/images/val"

    file_names = os.listdir(src_folder)
    random.shuffle(file_names)
    split_index = int(len(file_names) * 0.8)

    for i, file_name in enumerate(file_names):
        src_file = os.path.join(src_folder, file_name)
        if i < split_index:
            dst_file = os.path.join(dst_folder1, file_name)
        else:
            dst_file = os.path.join(dst_folder2, file_name)
        shutil.move(src_file, dst_file)

    for filename in os.listdir('/home/ma-user/modelarts/inputs/dataset/images/val'):
        shutil.move('/home/ma-user/modelarts/inputs/dataset/label/' + filename[:-4] + '.txt', '/home/ma-user/modelarts/inputs/dataset/labels/val')

    for filename in os.listdir('/home/ma-user/modelarts/inputs/dataset/label/'):
        shutil.move('/home/ma-user/modelarts/inputs/dataset/label/' + filename, '/home/ma-user/modelarts/inputs/dataset/labels/train')

