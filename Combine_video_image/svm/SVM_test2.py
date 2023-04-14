# 导入必要的库
import cv2
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
import time
import os
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC

data_dir = "data"

# 获取子目录列表
subdirs = ["eye_close", "eye_open"]

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

images = []
labels = []
for path in image_paths:
    image = cv2.imread(path, 0)  # 读取为灰度图像
    image = cv2.resize(image, (100,50))
    images.append(image)
    if "eye_close" in path:
        labels.append(-1)
    elif "eye_open" in path:
        labels.append(1)

# 提取HOG特征
features = []
for image in images:
    feature = hog(image, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(3, 3), visualize=False, transform_sqrt=True)
    features.append(feature)

scaler = StandardScaler()
features = scaler.fit_transform(features)

# X_test = features
# y_test = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 创建SVM对象
# Siz = len(y_test)
# print(Siz)
# clf = svm.SVC(kernel='linear', C=1.0)

# 定义SVM模型和参数搜索空间
svm = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# 使用GridSearchCV来寻找最佳参数组合
clf = GridSearchCV(svm, param_grid, cv=5)
# svm_grid.fit(train_data, train_labels)


# 训练SVM
clf.fit(X_train, y_train)
print("Best Parameters: ", clf.best_params_)

# 保存模型
joblib.dump(clf, 'svm_model.pkl')

clf = joblib.load('svm_model.pkl')
# 测试SVM
start_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
# print("time cost:",(end_time-start_time)/Siz)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cnt = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        cnt+=1
print(cnt)