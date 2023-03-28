# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import YawnDetectModule as yd

class bbox:
    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
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
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


    # 哈欠检测实例化
    yamn_detection = yd.YawnDetection('./mmod_human_face_detector.dat', './shape_predictor_68_face_landmarks.dat')

    #用于生成该视频所对应的编码
    seq = []
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            xfmin = 0
            yfmin = 0
            xfmax = 0
            yfmax = 0 # 面部的位置以及面部的坐标被存储
            nm = ""
            xpmin = 0
            ypmin = 0
            xpmax = 0
            ypmax = 0

            mouse = bbox(0, 0, 0, 0)
            eyes = bbox(0, 0, 0, 0)


            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                print(s)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # Results
                    xxmin = int(xyxy[0])
                    yymin = int(xyxy[1])
                    xxmax = int(xyxy[2])
                    yymax = int(xyxy[3])
                    nnm = names[int(cls)] #表示它的名字是什么
                    if(nnm == "mouse"):
                        if(xxmin + xxmax + yymax + yymin > mouse.xmax + mouse.xmin + mouse.ymax + mouse.ymin):
                            mouse.xmax = xxmax
                            mouse.xmin = xxmin
                            mouse.ymax = yymax
                            mouse.ymin = yymin

                    if(nnm == "eyes"):
                        if(xxmin + xxmax + yymax + yymin > eyes.xmax + eyes.xmin + eyes.ymax + eyes.ymin):
                            eyes.xmax = xxmax
                            eyes.xmin = xxmin
                            eyes.ymax = yymax
                            eyes.ymin = yymin


                    if (nnm == "Person" or nnm == "Sideface" or nnm == "Eyeclosing" or nnm == "Tophead"):
                        if(xxmin + xxmax + yymax + yymin > xfmax + xfmin + yfmax + yfmin):
                            xfmax = xxmax
                            xfmin = xxmin
                            yfmax = yymax
                            yfmin = yymin
                            nm = nnm
                    if(nnm == "Phone"):
                        if(xxmin + xxmax + yymax + yymin > xpmax + xpmin + ypmax + ypmin):
                            xpmax = xxmax
                            xpmin = xxmin
                            ypmax = yymax
                            ypmin = yymin

            # 循环结束之后会处理当前的最右下角的face以及最右下角的手机
            # 首先要判断是否检测到手机
            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            res_now = 0
            im0 = annotator.result()
            w, h = im0.shape[1], im0.shape[0]
            # xp = (xpmax + xpmin) / 2
            # yp = (ypmax + ypmin) / 2
            mouse_x = (mouse.xmax + mouse.xmin) / 2
            mouse_y = (mouse.ymax + mouse.ymin) / 2
            eyes_x = (eyes.xmax + eyes.xmin) / 2
            eyes_y = (eyes.ymax + eyes.ymin) / 2
            imaft = im0[yfmin:yfmax, xfmin:xfmax]

            flag_mouse = (mouse_x > w / 2 and mouse_y > h / 2)
            flag_eyes = (eyes_x > w / 2 and eyes_y > h / 2)

            if(flag_mouse):
                imaft_mouse = im0[mouse.ymin:mouse.xmax, mouse.xmin:mouse.xmax]
                # 在这里写预测函数——对应鼠标。注意，Imaft_mouse是一个numpy数组（frame）


            if(flag_eyes):
                imaft_eyes = im0[eyes.ymin:eyes.xmax, eyes.xmin:eyes.xmax]
                #在这里写预测函数——对应眼睛。注意，Imaft_mouse是一个numpy数组（frame）







            # if(xp > w / 2 and yp > h / 2):
            #     print("Calling!!!!!!!!!!!!!!!!!!!\n")
            #     seq.append(3)
            # elif(xfmax + xfmin > w):#一定检测到脸了
            #     if(nm == "Sideface"):#当前是侧脸
            #         print("Sideface!!!!!!!!!!!!!!!!!!!!!!!!!!\n") #当前在转头
            #         seq.append(4)
            #     elif(nm == "Eyeclosing" or nm == "Tophead" or nm == "Person"): # 表示当前检查司机的脸了
            #         gray = cv2.cvtColor(imaft, cv2.COLOR_BGR2GRAY) #转灰度图像
            #         res = yamn_detection.detect(gray)
            #         if(res == 0):
            #             print("!NORMAL")
            #             seq.append(0)
            #
            #         elif(res == 1):
            #             print("!EYECLOSING")
            #             seq.append(1)
            #         elif(res == 2):
            #             print("!YAMLING")
            #             seq.append(2)
            #         else:
            #             print("!WTF")
            #             seq.append(4) #当前已经检测不到人脸了，所以自动判别为左顾右盼
            #         # imag = im0[xfmin:xfmax, yfmin:yfmax]
            #         # cv2.imshow('Ans', imag)
            #     else:
            #         print("?")
            #         print(nm)
            #         seq.append(-1) # 异常状态检测
            # else:
            #     print("!NOT FOUND!") #此时司机的脸已经超出了屏幕的范围，直接定义为左顾右盼
            #     seq.append(4)
            # out_img = im0[:, :, [2, 1, 0]]

            # ros_image = out_img
            # cv2.imshow('YOLOV5', im0[:, :, [2, 1, 0]])

            # print(type(im0))
            # print(im0.shape)
            # print(xfmax)
            # print(yfmax)
            # imaft = im0[yfmin:yfmax, xfmin:xfmax]
            # res = yamn_detection.detect(imaft)

            # cv2.imshow('YOLOV5', imaft)





            # Stream results
            # im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), imaft.shape[1], imaft.shape[0])
                cv2.imshow(str(p), imaft)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    cntrgt = 0

    # for i in seq:
    #     if(i == int(list(source[-8])[0])):
    #         cntrgt += 1
    # #统计当前的视频里面有多少个cntrgt
    # print(cntrgt)
    # if(cntrgt >= 60):
    #     with open("res_all.txt", "a") as fc:
    #         fc.write('(')
    #         for i in seq:
    #             fc.write("%d" % i)
    #         fc.write(", %c)\n" % source[-8])
    #         fc.close()
    # else:
    #     with open("failed_path.txt", "a") as fai:
    #         fai.write("%s\n" % source)
    #         fai.close()



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)