## SVM
### 训练

Waiting to be updated...

-------------------

### SVM使用

```python
import svmdetect #在当前目录下

# 创建一个 ImageClassifier 实例
classifier = svmdetect.ImageClassifier()# 从文件中加载模型，如果没有模型文件则训练模型，括号里面是模型文件名路径

# 加载图像并提取特征
image_path = "./123.png"
images = classifier.load_images([image_path])
features = classifier.extract_features(images)

# 对特征进行分类并计算预测结果中负样本数量
y_pred, time_cost = classifier.classify(features)
cnt, ratio = classifier.count_negatives(y_pred)

# 输出结果
print("time cost:", time_cost)
print("pred:", y_pred)
print("negative count:", cnt)
print("negative ratio:", ratio)

```
-------------------

### 输出

-------------------

```
time cost: 0.0019681453704833984
pred: [1]
negative count: 0
negative ratio: 0.0
```