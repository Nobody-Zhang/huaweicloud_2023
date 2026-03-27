# coding=utf-8

import os
import requests

if __name__ == '__main__':
    # Config url, token and file path.
    url = os.environ.get("HUAWEICLOUD_MTCNN_URL", "")
    token = os.environ.get("HUAWEICLOUD_TOKEN", "")
    file_path = "../yolo/1.mp4"

    # Send request.
    headers = {
        'X-Auth-Token': token
    }
    files = {
        'images': open(file_path, 'rb')
    }
    resp = requests.post(url, headers=headers, files=files)

    # Print result.
    print(resp.status_code)
    print(resp.text)
