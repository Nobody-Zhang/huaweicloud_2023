# 导入必要的库
import cv2
from skimage.feature import hog
import time
import os
from sklearn.preprocessing import StandardScaler
import joblib

image_paths = ["./123.png"]
# data_dir = "data"
#
# # 获取子目录列表
# subdirs = ["eye_close"]
#
# # 创建一个空列表来存储所有图像的路径
# image_paths = []
#
# 循环遍历每个子目录
# for subdir in subdirs:
#     # 构造子目录的完整路径
#     subdir_path = os.path.join(data_dir, subdir)
#     # 循环遍历子目录下的所有图像文件
#     for filename in os.listdir(subdir_path):
#         # 构造图像文件的完整路径
#         image_path = os.path.join(subdir_path, filename)
#         # 将图像文件的路径添加到列表中
#         image_paths.append(image_path)
images = []

for path in image_paths:
    image = cv2.imread(path, 0)  # 读取为灰度图像
    image = cv2.resize(image, (60,36))
    images.append(image)


# 提取HOG特征
features = []
for image in images:
    feature = hog(image, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(3, 3), visualize=False, transform_sqrt=True)
    features.append(feature)

# scaler = StandardScaler()
# features = scaler.fit_transform(features)
features = [i/255 for i in features]
clf = joblib.load('svm_model.pkl')
# 测试SVM
start_time = time.time()
y_pred = clf.predict(features)
end_time = time.time()
print("time cost:",(end_time-start_time))
cnt = 0
print(len(y_pred))
for i in y_pred:
    if i == -1:
        cnt += 1
print(cnt)
print("pred:", y_pred)
print(cnt/len(y_pred))