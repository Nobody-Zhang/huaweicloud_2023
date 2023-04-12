import cv2
import os
import time
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import joblib

class ImageClassifier:
    def __init__(self, model_path='./svm_model.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = StandardScaler()

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            image = cv2.imread(path, 0)
            image = cv2.resize(image, (60, 36))
            images.append(image)
        return images

    def extract_features(self, images):
        features = []
        for image in images:
            feature = hog(image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(3, 3), visualize=False, transform_sqrt=True)
            features.append(feature)
        return self.scaler.fit_transform(features)

    def classify(self, features):
        features = [i / 255 for i in features]
        start_time = time.time()
        y_pred = self.model.predict(features)
        end_time = time.time()
        return y_pred, end_time - start_time

    def count_negatives(self, y_pred):
        cnt = 0
        for i in y_pred:
            if i == -1:
                cnt += 1
        return cnt, cnt / len(y_pred)
