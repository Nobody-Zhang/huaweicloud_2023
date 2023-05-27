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


def Sliding_Window(total_status, fps, thres=8 / 9):
    single_window_cnt = [0, 0, 0, 0, 0]
    # tmp = [0, 0, 0, 0, 0]
    # for i in range(len(total_status)):
    #     tmp[int(total_status[i])] += 1
    #
    # if tmp[3] >= int(thres * fps * 2):
    #     return 3
    threshold = int(thres * fps * 3)
    for i in range(len(total_status) - int(3 * fps)):
        if i == 0:
            for j in range(int(3 * fps)):
                # print(i + j)
                # print(tot_status[i + j])
                # print(type(tot_status[i + j]))
                single_window_cnt[int(total_status[i + j])] += 1
        else:
            single_window_cnt[int(total_status[i + int(3 * fps) - 1])] += 1
            single_window_cnt[int(total_status[i - 1])] -= 1
        for i in range(1, 5):
            if single_window_cnt[i] >= threshold:
                return i
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
def yolo_run(weights=ROOT / 'best_openvino_model/best.bin',  # model.pt path(s)
             source='',  # file/dir/URL/glob, 0 for webcam
             data=ROOT / 'one_stage.yaml',  # dataset.yaml path
             imgsz=(640, 640),  # inference size (height, width)
             conf_thres=0.20,  # confidence threshold
             iou_thres=0.40,  # NMS IOU threshold
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
    fps = dataset.cap.get(cv2.CAP_PROP_FPS)
    # FRAME_GROUP = int(fps / 3)
    # fps = 3

    cntt = 0
    tot_status = []
    YOLO_determin = YOLO_Status()
    # Run inference
    t_start = time_sync()  # start_time

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

    category = Sliding_Window(tot_status, fps)
    print(tot_status)
    # --------------------最后的返回！！！！！！-------------------------
    t_end = time_sync()  # end_time
    duration = t_end - t_start

    result = {"result": {"category": 0, "duration": 6000}}
    result['result']['category'] = category
    result['result']['duration'] = int(np.round((duration) * 1000))
    return fps, tot_status, category


if __name__ == "__main__":
    vidio_dir = "/home/hzkd/DATA/"
    save_dir = "/home/hzkd/Combine_video_image/Transformer_data/"
    i = 0
    loscnt = 0
    mixed = [[0, 0, 0, 0, 0] for i in range(5)]
    print(mixed)
    for fn in os.listdir(vidio_dir):
        i += 1
        print("cnt: " + str(i))
        print(fn)
        fn1 = vidio_dir + fn
        fps, tot_status, res = yolo_run(source=fn1)
        print("result: " + str(res))
        j = -1

        while (True):
            if fn1[j] == '_' or fn[j] == '-':
                break
            j -= 1
        right = fn1[j - 1]
        label = fn1[j - 2]
        if right == '1':  # 负样本
            label = str(0)
        fp = open(f"/home/hzkd/Combine_video_image/Transformer_data/{label}.txt", 'a')
        mixed[int(label)][res] += 1

        fp.write(fn + '\n' + str(fps) + ' ' + str(tot_status) + '\n')
        fp.close()
        if str(res) != str(label):
            fpf = open(f"/home/hzkd/Combine_video_image/Transformer_data/failed.txt", 'a')
            fpf.write("file_name: " + fn + '\n' + "right_status: " + label + '\n' + "predic_status:" + str(
                res) + '\n' + "FPS and failed tot_status:\n" + str(fps) + ' ' + str(tot_status) + '\n')
            loscnt += 1
        print("tot: " + str(i))
        print("loss_cnt: " + str(loscnt))
        print("percent: " + str(loscnt / i))
        for w in range(5):
            print(mixed[w])

    # print(result)