# coding=utf-8

import logging
import os

import requests

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Config url, token and file path.
    url = os.environ.get("HUAWEICLOUD_MTCNN_URL", "")
    token = os.environ.get("HUAWEICLOUD_TOKEN", "")
    file_path = "../yolo/1.mp4"

    # Send request.
    headers = {"X-Auth-Token": token}
    files = {"images": open(file_path, "rb")}
    resp = requests.post(url, headers=headers, files=files)

    # Log result.
    logger.debug("status_code: %s", resp.status_code)
    logger.debug("response: %s", resp.text)
