# 用训练好的nanodet模型对视频/图片进行划分
# Predictor.finding_face() 整张图找脸
# Predictor.finding_eyes()、Predictor.finding_mouth() 脸找眼睛和嘴巴
# 用例：
# meta, res = predictor.inference(frame)
# imaft = predictor.finding_face(res,meta,0.75)
import argparse
import os
import time
import platform
from pathlib import Path
import cv2
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, load_config, load_model_weight
from nanodet.util import cfg1 as cfg
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", default="./config/nanodet-plus-m_416.yml", help="model config file path")
    parser.add_argument("--model", default="./nanodet-plus-m_416.pth",help="model file path")
    parser.add_argument("--path", default="./day_man_002_20_1.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cpu"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img

    def finding_face(self, dets, meta,score_thres):
        # 处理所有的class，返回所有的box，包括label，score
        all_box = []
        for k1, v1 in dets.items():
            for label, v2 in v1.items():
                if len(v2) > 0:
                    for i in range(len(v2)):
                        score = v2[i][-1]
                        if score > score_thres:
                            x0, y0, x1, y1 = [int(i) for i in v2[i][:4]]
                            all_box.append([label, x0, y0, x1, y1, score])
        # 按照score排序，然后遍历box里面的信息，裁剪图片
        all_box.sort(key=lambda v: v[5])
        # 读取原始图片
        im0 = meta["raw_img"][0]
        imaft = im0
        for box in all_box:
            label, x0, y0, x1, y1, score = box
            if label == 0:
                imaft = im0[y0:y1, x0:x1] # 但愿这里的坐标是正确的
        return imaft
    
    def finding_eyes(self, dets, meta,score_thres):
        # 处理所有的class，返回所有的box，包括label，score
        all_box = []
        for k1, v1 in dets.items():
            for label, v2 in v1.items():
                if len(v2) > 0:
                    for i in range(len(v2)):
                        score = v2[i][-1]
                        if score > score_thres:
                            x0, y0, x1, y1 = [int(i) for i in v2[i][:4]]
                            all_box.append([label, x0, y0, x1, y1, score])
        # 按照score排序，然后遍历box里面的信息，裁剪图片
        all_box.sort(key=lambda v: v[5])
        # 读取原始图片
        im0 = meta["raw_img"][0]
        for box in all_box:
            label, x0, y0, x1, y1, score = box
            if label == 0:
                imaft = im0[y0:y1, x0:x1] # 但愿这里的坐标是正确的
        return imaft
    
    def finding_mouth(self, dets, meta,score_thres):
        # 处理所有的class，返回所有的box，包括label，score
        all_box = []
        for k1, v1 in dets.items():
            for label, v2 in v1.items():
                if len(v2) > 0:
                    for i in range(len(v2)):
                        score = v2[i][-1]
                        if score > score_thres:
                            x0, y0, x1, y1 = [int(i) for i in v2[i][:4]]
                            all_box.append([label, x0, y0, x1, y1, score])
        # 按照score排序，然后遍历box里面的信息，裁剪图片
        all_box.sort(key=lambda v: v[5])
        # 读取原始图片
        im0 = meta["raw_img"][0]
        imaft = im0
        for box in all_box:
            label, x0, y0, x1, y1, score = box
            if label == 1:
                imaft = im0[y0:y1, x0:x1] # 但愿这里的坐标是正确的
                print("find mouth")
        return imaft


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cpu")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        # save_folder = os.path.join(
        #     cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        # )
        # mkdir(local_rank, save_folder)
        # save_path = (
        #     os.path.join(save_folder, args.path.replace("\\", "/").split("/")[-1])
        #     if args.demo == "video"
        #     else os.path.join(save_folder, "camera.mp4")
        # )
        # print(f"save_path is {save_path}")
        # vid_writer = cv2.VideoWriter(
        #     save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        # )
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame)
                # print("\ndets:")
                # print(res)
                # result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                #这里进行单独的裁剪框
                imaft = predictor.finding_face(res,meta,0.75)
                # print("imaft.shape:")
                # print(imaft.shape)
                # cv2.namedWindow('face', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                # cv2.resizeWindow('face', imaft.shape[1], imaft.shape[0])
                # cv2.imshow('face', imaft)
                # cv2.waitKey(1)  # 1 millisecond
                # if args.save_result:
                #     vid_writer.write(result_frame)
                # ch = cv2.waitKey(1)
                # if ch == 27 or ch == ord("q") or ch == ord("Q"):
                #     break
            else:
                break


if __name__ == "__main__":
    main()
