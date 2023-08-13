import time
import cv2
import torch
from model_service.pytorch_model_service import PTServingBaseService
import sys
from pathlib import Path
import warnings
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        print(device)
        # device = 'cuda:0'
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def get_rotation(points):
    """
    Parameters
    ----------
    points : float32, Size = (10,)
        coordinates of landmarks for the selected faces.
    Returns
    -------
    roll    , yaw   , pitch
    float32, float32, float32
    """

    LMx = points[0:5]  # horizontal coordinates of landmarks
    LMy = points[5:10] # vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
    
    alpha = np.cos(angle)
    beta = np.sin(angle)
    
    print(LMx,LMy)
    # rotated landmarks
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    
    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    
    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    
    # relative rotation 0 degree is frontal 90 degree is profile
    Xfrontal = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return angle * 180 / np.pi, Xfrontal, Yfrontal

class retinaface_model:
    def __init__(self):
        self.weights = ROOT / "weights/mobilenet0.25_Final.pth"
        self.network = "mobile0.25"  # Backbone network mobile0.25 or resnet50
        self.cpu = True  # Use cpu/GPU inference
        self.confidence_threshold = 0.02  # confidence_threshold
        self.top_k = 5000  # top_k
        self.nms_threshold = 0.4  # nms_threshold
        self.keep_top_k = 750  # keep_top_k
        self.vis_thres = 0.8  # visualization_threshold
        self.cfg = cfg_mnet
        torch.set_grad_enabled(False)
        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network == "resnet50":
            self.cfg = cfg_re50
        
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, self.weights, self.cpu)
        self.net.eval()
        print('Finished loading model!')
        # print(net)
        
    def inference(self, source):
        tic = time.time()
        self.video_path = ROOT / str(source)
        cudnn.benchmark = True
        if self.cpu:
            device = torch.device("cpu")
        else:
            device = torch.cuda.current_device()
        print(device)
        self.net = self.net.to(device)
        resize = 1

        image_path = str(self.video_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        # print('input shape ', img.shape)
        # tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, nms_threshold,force_cpu=cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        print(dets)
        for b in dets:
            if b[4] < self.vis_thres:
                continue

            bb = np.array(b)

            # x\y
            point = []
            for i in range(5, 15):
                if i % 2 == 0:
                    point.append(bb[i])  # point[0/2/4/6/8] is x

            for i in range(5, 15):
                if i % 2 == 1:
                    point.append(bb[i])  # point[1/3/5/7/9] is y

            point = np.transpose(point)
            angle, Xfrontal, Yfrontal = get_rotation(point)

            print("Roll: {0:.2f} degrees".format(angle))
            print("Yaw: {0:.2f} degrees".format(Xfrontal))
            print("Pitch: {0:.2f} degrees".format(Yfrontal))

            result = {"result": {"Roll": 0.0, "Yaw": 0.0, "Pitch": 0.0, "duration": 6000}}

            result['result']['Roll'] = angle
            result['result']['Yaw'] = Xfrontal
            result['result']['Pitch'] = Yfrontal
            result['result']['duration'] = int((time.time() - tic) * 1000)
        
        return result


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        # super(PTVisionService, self).__init__(model_name, model_path)
        # 调用自定义函数加载模型
        self.capture = 'test.jpg'
        self.model_name = model_name
        self.model_path = model_path
        self.model = retinaface_model()

    def _inference(self, data):
        result = self.model.inference(source = self.capture)
        # result = nanodet_run(source = self.capture)
        return result

    def _preprocess(self, data):
        # 这个函数把data写到test.mp4里面了
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _postprocess(self, data):
        return data
