# 导入必要的库
import cv2
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import time
import os
from sklearn.preprocessing import StandardScaler
import joblib

data_dir = "archive1"

# 获取子目录列表
subdirs = ["no yawn", "yawn"]

# 创建一个空列表来存储所有图像的路径
image_paths = []

# 循环遍历每个子目录
for subdir in subdirs:
    # 构造子目录的完整路径
    subdir_path = os.path.join(data_dir, subdir)
    # 循环遍历子目录下的所有图像文件
    for filename in os.listdir(subdir_path):
        # 构造图像文件的完整路径
        image_path = os.path.join(subdir_path, filename)
        # 将图像文件的路径添加到列表中
        image_paths.append(image_path)

# 打印所有图像的路径列表
# print(image_paths)
images = []
labels = []
for path in image_paths:
    image = cv2.imread(path, 0)  # 读取为灰度图像
    image = cv2.resize(image, (105,59))
    images.append(image)
    if "eye_close" in path:
        labels.append(0)
    elif "eye_open" in path:
        labels.append(1)
    # 其他类别的标签...

# 提取HOG特征
features = []
for image in images:
    feature = hog(image, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(3, 3), visualize=False, transform_sqrt=True)
    features.append(feature)

scaler = StandardScaler()
features = scaler.fit_transform(features)

# l = feature[0].size
# for feature in features:
#     if feature.size != l:
#         print("error!!!!!!!!!!!")
#         break

# features = np.array(features)
# print(features)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# print(X_train)
# 创建SVM对象
Siz = len(y_test)
print(Siz)
clf = svm.SVC(kernel='rbf', C=1.0)

# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# 训练SVM
clf.fit(X_train, y_train)

# 保存模型
joblib.dump(clf, 'svm_model.pkl')

clf = joblib.load('svm_model.pkl')
# 测试SVM
start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
print("time cost:",(end_time-start_time)/Siz)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)