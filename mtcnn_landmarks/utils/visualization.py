import cv2
import os
import numpy as np
# 画出人脸框和关键点
def draw_face(image_path, boxes_c, landmarks,output_path):
    img = cv2.imread(image_path)
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        # 判别为人脸的置信度
        cv2.putText(img, '{:.2f}'.format(score),
                    (corpbbox[0], corpbbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # 画关键点
    for i in range(landmarks.shape[0]):
        for j in range(len(landmarks[i]) // 2):
            cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    # 将img保存到output_path文件夹下的同名图片中
    cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), img)


def draw_face_video(img, boxes_c, landmarks):
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
    # 画关键点
    for i in range(landmarks.shape[0]):
        for j in range(len(landmarks[i]) // 2):
            cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))


def draw_euler_angles(image, euler_angles, axis_length, points):
    angles_degrees = euler_angles
    cv2.putText(image, "roll: {:.2f}".format(angles_degrees[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "yaw: {:.2f}".format(angles_degrees[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, "pitch: {:.2f}".format(angles_degrees[2]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # 将关键点整合为(x,y)
    landmarks = np.array([(x, y) for x, y in zip([points[0], points[2], points[4], points[6], points[8]],
                                                 [points[1], points[3], points[5], points[7], points[9]])])
    # 鼻子作为原点，将2维坐标转换为3维坐标
    origin = np.expand_dims(landmarks[2], axis=0)
    # 计算旋转矩阵
    rotation_matrix = cv2.Rodrigues(np.radians(euler_angles))[0]

    # 绘制坐标系
    x_axis_end = (origin + axis_length * rotation_matrix[0][:2]).astype(int)
    y_axis_end = (origin + axis_length * rotation_matrix[1][:2]).astype(int)
    z_axis_end = (origin + axis_length * rotation_matrix[2][:2]).astype(int)

    # 将origin, x_axis_end, y_axis_end, z_axis_end转换为tuple
    origin = tuple(origin[0].astype(int))
    x_axis_end = tuple(x_axis_end[0])
    y_axis_end = tuple(y_axis_end[0])
    z_axis_end = tuple(z_axis_end[0])

    cv2.line(image, origin, x_axis_end, (0, 0, 255), 2)  # 红色 x 轴
    cv2.line(image, origin, y_axis_end, (0, 255, 0), 2)  # 绿色 y 轴
    cv2.line(image, origin, z_axis_end, (255, 0, 0), 2)  # 蓝色 z 轴

    return image


def draw_coordinate_system(image, origin, axis_length=100):
    x_axis_end = (origin[0] + axis_length, origin[1])
    y_axis_end = (origin[0], origin[1] + axis_length)
    z_axis_end = (origin[0], origin[1] - axis_length)

    cv2.line(image, origin, x_axis_end, (0, 0, 255), 2)  # 红色 x 轴
    cv2.line(image, origin, y_axis_end, (0, 255, 0), 2)  # 绿色 y 轴
    cv2.line(image, origin, z_axis_end, (255, 0, 0), 2)  # 蓝色 z 轴