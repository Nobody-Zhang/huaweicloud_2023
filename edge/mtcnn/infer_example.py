from infer_API import infer_video, infer_image,load_models
import torch
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 0. 加载模型
    model_path = './infer_models'
    device = torch.device("cpu")
    load_models(model_path,device)

    # 1. 图片推理
    image_path = './test/1.jpg'
    res = infer_image(image_path,device)
    logger.info("res: %s", res)

    # 2. 视频推理
    video_path = './test/1.mp4'
    res = infer_video(video_path,device)
    logger.info("res: %s", res)