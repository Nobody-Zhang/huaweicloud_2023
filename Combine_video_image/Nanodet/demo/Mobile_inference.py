import os
import sys
import json
import time

from torch import nn
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
import numpy as np

def _make_divisible(ch, divisor=8, min_ch=None):
     """
     This function is taken from the original tf repo.
     It ensures that all layers have a channel number that is divisible by 8
     It can be seen here:
     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
     """
     if min_ch is None:
          min_ch = divisor
     new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
     # Make sure that round down does not go down by more than 10%.
     if new_ch < 0.9 * ch:
          new_ch += divisor
     return new_ch


class ConvBNReLU(nn.Sequential):
     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
          padding = (kernel_size - 1) // 2
          super(ConvBNReLU, self).__init__(
               nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
               nn.BatchNorm2d(out_channel),
               nn.ReLU6(inplace=True)
          )


class InvertedResidual(nn.Module):
     def __init__(self, in_channel, out_channel, stride, expand_ratio):
          super(InvertedResidual, self).__init__()
          hidden_channel = in_channel * expand_ratio
          self.use_shortcut = stride == 1 and in_channel == out_channel

          layers = []
          if expand_ratio != 1:
               # 1x1 pointwise conv
               layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
          layers.extend([
               # 3x3 depthwise conv
               ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
               # 1x1 pointwise conv(linear)
               nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
               nn.BatchNorm2d(out_channel),
          ])

          self.conv = nn.Sequential(*layers)

     def forward(self, x):
          if self.use_shortcut:
               return x + self.conv(x)
          else:
               return self.conv(x)


class MobileNetV2(nn.Module):
     def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
          super(MobileNetV2, self).__init__()
          block = InvertedResidual
          input_channel = _make_divisible(32 * alpha, round_nearest)
          last_channel = _make_divisible(1280 * alpha, round_nearest)

          inverted_residual_setting = [
               # t, c, n, s
               [1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 4, 2],
               [6, 96, 3, 1],
               [6, 160, 3, 2],
               [6, 320, 1, 1],
          ]

          features = []
          # conv1 layer
          features.append(ConvBNReLU(1, input_channel, stride=2))
          # building inverted residual residual blockes
          for t, c, n, s in inverted_residual_setting:
               output_channel = _make_divisible(c * alpha, round_nearest)
               for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
          # building last several layers
          features.append(ConvBNReLU(input_channel, last_channel, 1))
          # combine feature layers
          self.features = nn.Sequential(*features)

          # building classifier
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          self.classifier = nn.Sequential(
               nn.Dropout(0.2),
               nn.Linear(last_channel, num_classes)
          )

          # weight initialization
          for m in self.modules():
               if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                         nn.init.zeros_(m.bias)
               elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
               elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

     def forward(self, x):
          x = self.features(x)
          x = self.avgpool(x)
          x = torch.flatten(x, 1)
          x = self.classifier(x)
          return x

class Inference():
     def __init__(self, weight_path="./MobileNetV2_4class.pth",device=torch.device("cpu")):
          # create model
          self.model = MobileNetV2(num_classes=4).to(device)
          # load model weights
          self.model_weight_path = weight_path
          self.device = device
          self.model.load_state_dict(torch.load(self.model_weight_path, map_location=device))
          self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
          #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          #   transforms.Normalize([0.5], [0.5])
            transforms.Normalize([0.449], [0.226])
          ])
          # read class_indict
          self.json_path = "4class_indices.json"
          with open(self.json_path, "r") as f:
               self.class_indict = json.load(f)
          self.model = self.model.eval()
     
     def img_path_transform(self,img_path):
          img = Image.open(img_path)
          # img = img.convert('L')
          # convert to 3 channel (RGB)
          # img = np.stack((img,)*3, axis=-1)
          img = self.transform(img)
          # expand batch dimension
          img = torch.unsqueeze(img, dim=0)
          return img
     
     def img_transform(self,img):
          image = Image.fromarray(img)
          image = self.transform(image)
          image = torch.unsqueeze(image, dim=0)
          return image
     
     def inference(self,img):
          img = self.img_transform(img)
          # predict class
          with torch.no_grad():
               output = torch.squeeze(self.model(img.to(self.device))).cpu()
               predict = torch.softmax(output, dim=0)
               predict_cla = torch.argmax(predict).numpy()
          
          return predict_cla
     
     def inference_imgpath(self,imgpath):
          img = self.img_path_transform(imgpath)
          # predict class
          with torch.no_grad():
               output = torch.squeeze(self.model(img.to(self.device))).cpu()
               predict = torch.softmax(output, dim=0)
               predict_cla = torch.argmax(predict).numpy()
          
          return predict_cla

def example_use_cpu():
     model = Inference()
     # get imgs path
     # img_path = "../train-data/pics/wjt_cut/yawn_mouth"

     img = "./open_eye.jpg"
     

     start = time.time()
     inference_class = model.inference_imgpath(img)
     end = time.time()

     cost = end - start
     print("This picture belongs to class : {} cost : {}".format(inference_class,cost))


def example_use_gpu():
     # add device define
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     model = Inference(device = device)
     # get imgs path
     # img_path = "../train-data/pics/Eye_train/eye_open"
     # img_path = "../train-data/pics/four_classes2/train/not_yawn"
     img_path = "../train-data/pics/wjt_cut/yawn_mouth"
     imgs = [os.path.join(img_path, f) for f in os.listdir(img_path)]
     imgs_num = len(imgs)

     count = 0
     min_cost = 99999.9
     max_cost = 0.0
     avg_cost = 0

     for img in imgs:
          start = time.time()
          inference_class = model.inference_imgpath(img)
          end = time.time()
          if inference_class == 3:
               count += 1
          print("This picture belongs to class : {}".format(inference_class))
          cost = end - start
          avg_cost += cost
          min_cost = min(min_cost , cost)
          max_cost = max(max_cost , cost)

     print("GPU_EXAMPLE--acc : {:.3}  min cost : {}  max cost : {}  avg cost:{}".format(count * 1.0 / imgs_num * 1.0 , min_cost,max_cost,avg_cost / imgs_num*1.0))

if __name__ == '__main__':
     # example_use_cpu()
     example_use_gpu()