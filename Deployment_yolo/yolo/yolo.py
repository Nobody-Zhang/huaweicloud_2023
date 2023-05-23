# import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import *
from utils.torch_utils import select_device, time_sync


# write by llr
# transform xyxy loacationn to xywh loacation, scale in (0, 1)
def xyxy2xywh(xmin: int, ymin: int, xmax: int, ymax: int, wide: int, height: int) -> tuple:
    """
    tranform xyxy location to xywh location

    :param xmin: xmin
    :param ymin: ymin
    :param xmax: xmax
    :param ymax: ymax
    :param wide: wide
    :param height: height
    :return: tuple(x,y,w,h)
    """
    x = ((xmin + xmax) // 2) / wide
    y = ((ymin + ymax) // 2) / height
    w = (xmax - xmin) / wide
    h = (ymax - ymin) / height
    return x, y, w, h


def Sliding_Window(tot_status, fps, thres1=2.48, thres2=0.48):
    window_status = {}  # 所有窗口的状态
    window_status_cnt = [0, 0, 0, 0, 0]
    single_window_cnt = [0, 0, 0, 0, 0]
    """
    window_status_cnt = {}  # 窗口状态计数
    window_status_cnt[0] = 0
    window_status_cnt[1] = 0
    window_status_cnt[2] = 0
    window_status_cnt[3] = 0
    window_status_cnt[4] = 0
    single_window_cnt = {}
    single_window_cnt[0] = 0
    single_window_cnt[1] = 0
    single_window_cnt[2] = 0
    single_window_cnt[3] = 0
    single_window_cnt[4] = 0
    """
    for i in range(len(tot_status) - int(2.5 * fps)):
        if i == 0:
            for j in range(int(2.5 * fps)):
                print(i + j)
                print(tot_status[i + j])
                print(type(tot_status[i + j]))
                single_window_cnt[int(tot_status[i + j])] += 1
        else:
            single_window_cnt[int(tot_status[i + int(2.5 * fps) - 1])] += 1
            single_window_cnt[int(tot_status[i - 1])] -= 1
        single_window_cnt[0] = -1  # 排除0
        max_cnt = 0

        # max_cnt = max(single_window_cnt, key=lambda x: single_window_cnt[x])
        for j in range(len(single_window_cnt)):
            if single_window_cnt[j] > single_window_cnt[max_cnt]:
                max_cnt = j
        if single_window_cnt[max_cnt] >= thres1 * fps:
            window_status[i] = max_cnt
        else:
            window_status[i] = 0
    for i in range(len(window_status)):
        window_status_cnt[int(window_status[i])] += 1
    print("window_status:", window_status)
    print("window_status_cnt:", window_status_cnt)
    window_status_cnt[0] = -1  # 排除0
    max_status = 0
    for i in range(len(window_status_cnt)):
        if (window_status_cnt[max_status] < window_status_cnt[i]):
            max_status = i
    # max_status = max(window_status_cnt, key=lambda x: window_status_cnt[x])
    if window_status_cnt[max_status] >= thres2 * fps:
        return max_status
    else:
        return 0


class YOLO_Status:
    def __init__(self):
        self.cls_ = {"close_eye": 0, "close_mouth": 1, "face": 2, "open_eye": 3, "open_mouth": 4, "phone": 5,
                     "sideface": 6}
        self.status_prior = {"normal": 0, "closeeye": 1, "yawn": 2, "calling": 4, "turning": 3}
        self.condition = [0, 1, 2, 4, 3]
        pass

    def determin(self, img, dets) -> int:
        """
        determin which status this frame belongs to\n
        0 -> normal status\n
        1 -> close eye\n
        2 -> yawn\n
        3 -> calling\n
        4 -> turning around\n

        :param img: input image, format the same as detect function
        :param dets: the detect boxes
        :returns: an int status symbol
        """
        wide, height = img.shape[1], img.shape[0]  # 输入图片宽、高
        status = 0  # 最终状态，默认为0
        driver = (0, 0, 0, 0)  # 司机正脸xywh坐标
        driver_conf = 0  # 正脸可信度
        sideface = (0, 0, 0, 0)  # 司机侧脸xywh坐标
        sideface_conf = 0  # 侧脸可信度
        phone = (0, 0, 0, 0)  # 手机xywh坐标
        openeye = (0, 0, 0, 0)  # 睁眼xywh坐标
        closeeye = (0, 0, 0, 0)  # 闭眼xywh坐标， 以防两只眼睛识别不一样
        openeye_score = 0  # 睁眼可信度
        closeeye_score = 0  # 闭眼可信度
        mouth = (0, 0, 0, 0)  # 嘴xywh坐标
        mouth_status = 0  # 嘴状态，0 为闭， 1为张

        # 处理boxes
        bboxes = dets
        for box in bboxes:  # 遍历每个box
            xyxy = tuple(box[:4])  # xyxy坐标
            xywh = xyxy2xywh(*xyxy, wide, height)  # xywh坐标
            conf = box[4]  # 可信度
            cls = box[5]  # 类别
            if cls == self.cls_["face"]:  # 正脸
                if .5 < xywh[0] and xywh[1] > driver[1] and xyxy[3] / height > .25:
                    # box中心在右侧0.5 并且 在司机下侧 并且 右下角在y轴0.25以下
                    driver = xywh  # 替换司机
                    driver_conf = conf
            elif cls == self.cls_["sideface"]:  # 侧脸
                if .5 < xywh[0] and xywh[1] > sideface[1] and xyxy[3] / height > .25:  # box位置，与face一致
                    sideface = xywh  # 替换侧脸
                    sideface_conf = conf
            elif cls == self.cls_["phone"]:  # 手机
                if .4 < xywh[0] and .2 < xywh[1] and xywh[1] > phone[1] and xywh[0] > phone[0]:
                    # box位置在右0.4, 下0.2, 原手机右下
                    phone = xywh  # 替换手机
            elif cls == self.cls_["open_eye"]:  # 睁眼
                if xywh[0] > openeye[0]:  # 找最右边的，下面的同理
                    openeye = xywh
                    openeye_score = conf
            elif cls == self.cls_["close_eye"]:  # 闭眼
                if xywh[0] > closeeye[0]:
                    closeeye = xywh
                    closeeye_score = conf
            elif cls == self.cls_["open_mouth"]:  # 张嘴
                if xywh[0] > mouth[0]:
                    mouth = xywh
                    mouth_status = 1
            elif cls == self.cls_["close_mouth"]:  # 闭嘴
                if xywh[0] > mouth[0]:
                    mouth = xywh
                    mouth_status = 0

        # 判断状态
        if driver[0] > sideface[0] and 0 < abs(driver[0] - phone[0]) < .3 and 0 < abs(
                driver[1] - phone[1]) < .3:  # 正脸打电话，手机与正脸相对0～0.3之内
            status = max(status, self.status_prior["calling"])  # 判断状态为打电话
        elif driver[0] < sideface[0] and 0 < abs(sideface[0] - phone[0]) < .3 and 0 < abs(
                sideface[1] - phone[1]) < .3:  # 侧脸打电话，同正脸判断
            status = max(status, self.status_prior["calling"])

        if abs(driver[0] - sideface[0]) < .1 and abs(driver[1] - sideface[1]) < .1:  # 正脸与侧脸很接近，说明同时检测出了正脸和侧脸
            if driver_conf > sideface_conf:  # 正脸可信度更高
                status = max(status, self.status_prior["normal"])
            else:  # 侧脸可信度更高
                status = max(status, self.status_prior["turning"])
        elif sideface[0] > driver[0]:  # 正侧脸不重合，并且侧脸在正脸右侧，说明司机是侧脸
            status = max(status, self.status_prior["turning"])

        if mouth_status == 1:  # 嘴是张着的
            status = max(status, self.status_prior["yawn"])

        if abs(closeeye[0] - openeye[0]) < .2:  # 睁眼和闭眼离得很近， 说明是同一个人两只眼睛判断得不一样
            if closeeye_score > openeye_score:  # 闭眼可信度比睁眼高
                status = max(status, self.status_prior["closeeye"])
            else:
                status = max(status, self.status_prior["normal"])
        else:  # 说明是两个人的眼睛，靠右边的是司机的眼睛
            if closeeye[0] > openeye[0]:  # 司机是闭眼
                status = max(status, self.status_prior["closeeye"])
            else:  # 司机是睁眼
                status = max(status, self.status_prior["normal"])

        return self.condition[status]


@torch.no_grad()
def yolo_run(weights=ROOT / 'INT8_openvino_model/best_int8.xml',  # model.pt path(s)
             source='',  # file/dir/URL/glob, 0 for webcam
             data=ROOT / 'one_stage.yaml',  # dataset.yaml path
             imgsz=(640, 640),  # inference size (height, width)
             conf_thres=0.25,  # confidence threshold
             iou_thres=0.45,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             view_img=False,  # show results
             save_txt=False,  # save results to *.txt
             save_conf=False,  # save confidences in --save-txt labels
             save_crop=False,  # save cropped prediction boxes
             nosave=False,  # do not save images/videos
             classes=None,  # filter by class: --class 0, or --class 0 2 3
             agnostic_nms=False,  # class-agnostic NMS
             augment=False,  # augmented inference
             visualize=False,  # visualize features
             update=False,  # update all models
             project=ROOT / 'runs/detect',  # save results to project/name
             name='exp',  # save results to project/name
             exist_ok=False,  # existing project/name ok, do not increment
             line_thickness=3,  # bounding box thickness (pixels)
             hide_labels=False,  # hide labels
             hide_conf=False,  # hide confidences
             half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             FRAME_GROUP=1,  # frame group
             ):
    source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    bs = 1  # batch_size
    # Dataloader

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # Run inference

    t_start = time_sync()  # start_time

    cntt = 0
    tot_status = []
    YOLO_determin = YOLO_Status()

    # ---------------------------------------------
    for path, im, im0s, vid_cap, s in dataset:
        cntt += 1
        if cntt % FRAME_GROUP != 0:
            continue  # skip some frames
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                print(det.numpy())
                cur_status = YOLO_determin.determin(im0, det.numpy())
                tot_status.append(cur_status)

                print(cur_status)

        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    # -------------------一定注意，这里得到的是tot_status，be like [0, 0, 2, ...]，数字！--------------------------
    fps = int(dataset.cap.get(cv2.CAP_PROP_FRAME_COUNT) / FRAME_GROUP)
    category = Sliding_Window(tot_status, fps)

    # --------------------最后的返回！！！！！！-------------------------
    t_end = time_sync()  # end_time
    duration = t_end - t_start

    result = {"result": {"category": 0, "duration": 6000}}
    result['result']['category'] = category
    result['result']['duration'] = int(np.round((duration) * 1000))
    return result


if __name__ == "__main__":
    result = yolo_run(source='night_woman_005_31_4.mp4')
    print(result)