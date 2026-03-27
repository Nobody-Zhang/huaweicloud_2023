# coding=utf-8
from __future__ import annotations

import requests
from apig_sdk import signer
import os
from obs import ObsClient
import random
import string
import time
from typing import Tuple

from obs import DeleteObjectsRequest, Object


sig = signer.Signer()
# Set the AK/SK to sign and authenticate the request.
access_key = os.environ.get("HUAWEICLOUD_AK")
secret_key = os.environ.get("HUAWEICLOUD_SK")
sig.Key = access_key
sig.Secret = secret_key
ima_id = os.environ.get("HUAWEICLOUD_IMA_ID", "")
#训练作业URL
train_job_url = "https://modelarts.cn-north-4.myhuaweicloud.com/v2/"+ima_id+"/training-jobs"

obs_url = 'https://obs.cn-north-4.myhuaweicloud.com'


obs_client = ObsClient(

    access_key_id=access_key,
    secret_access_key=secret_key,
    server=obs_url
)

bucket_name = 'auto-train-jyf'
upload_dir = 'yoloV5/input/'

local_file_path = './auto-train/test.png'
obs_file_name = 'atest.png'

# 视频文件的log日志
log_dir = "../ota_test/smart_record.log"
# 触发在线训练的阈值
MP4_threshold = 3

#设置训练代码在obs中的目录和启动代码
code_dir = "/auto-train-jyf/yoloV5/yolov5/"
boot_file = "/auto-train-jyf/yoloV5/yolov5/train.py"
#数据集在obs中的位置，也是上传的位置
input_dir = "/" + bucket_name + "/" + upload_dir
#训练结果在obs中的位置，也是要下载的位置
#注意，在训练结束后，需要执行类似os.system("cp -r ./runs/train/output/* ../../output/")的命令，前者是训练结果存放的目录，一般为./runs/train/exp,后者不需要改动
output_dir = "/auto-train-jyf/yoloV5/output/"


def random_str(n: int) -> str:
    """Generate a random alphanumeric string of specified length.

    Args:
        n: Desired length of the random string. Must not exceed the
            size of the character pool (62 characters).

    Returns:
        A random string of length ``n`` composed of ASCII letters
        (upper and lower) and digits.
    """
    s = string.ascii_letters + string.ascii_uppercase + string.digits
    return ''.join(random.sample(s, n))

def upload_file_to_obs(bucket_name: str, local_file_path: str, obs_file_name: str) -> None:
    """Upload a local file to Huawei Cloud OBS (Object Storage Service).

    Args:
        bucket_name: Name of the target OBS bucket.
        local_file_path: Path to the local file to upload.
        obs_file_name: Destination object key (path) in OBS.

    Raises:
        Exception: Prints the error message if the upload fails.
    """
    try:
        # upload train data
        obs_client.putFile(bucket_name, obs_file_name, local_file_path)
        print("upload success!")
    except Exception as e:
        print("upload failed: ", e)


def check_job_status(job_id: str) -> bool:
    """Check the status of a ModelArts training job.

    Queries the ModelArts training job API to determine whether the
    specified job has completed.

    Args:
        job_id: The unique identifier of the training job.

    Returns:
        True if the training job has completed, False otherwise.
    """
    job_url = train_job_url+"/"+job_id
    payload = ""
    r = signer.HttpRequest("GET",job_url)
    r.headers = {"content-type": "application/json"}
    r.body = payload
    sig.Sign(r)
    response = requests.request(r.method, r.scheme + "://" + r.host + r.uri, headers=r.headers, data=r.body)
    resp_json = response.json()
    status = resp_json["status"]["secondary_phase"]
    print("training job status: "+status)
    if status == "Completed":
        return True
    return False

def create_training_job() -> Tuple[int, str]:
    """Create a new ModelArts training job via the REST API.

    Submits a training job to ModelArts using PyTorch 1.8.0 engine with
    the code, dataset, and output paths configured at module level. The
    job uses the free-tier GPU flavor ``modelarts.p3.large.public.free``.

    Returns:
        A tuple of ``(status_code, job_id)`` where *status_code* is the
        HTTP response code (201 on success) and *job_id* is the unique
        identifier assigned to the created training job.
    """
    r = signer.HttpRequest("POST", train_job_url)
    r.headers = {"content-type": "application/json"}
    job_name = "job-"+random_str(8)
    #训练的引擎是固定的，PyTorch只能采用python3.7，因此需要确认训练代码是否支持3.7
    #所有训练的参数请在代码中固定
    payload = "{\"kind\":\"job\",\"metadata\":{\"name\":"+"\""+job_name+"\""+",\"workspace_id\":\"0\",\"description\":\"This is a ModelArts job\"},\"algorithm\":{\"code_dir\":"+"\""+code_dir+"\""+",\"boot_file\":"+"\""+boot_file+"\""+",\"parameters\":[],\"inputs\":[{\"name\":\"data_url\",\"remote\":{\"obs\":{\"obs_url\":"+"\""+input_dir+"\""+"}}}],\"outputs\":[{\"name\":\"train_url\",\"remote\":{\"obs\":{\"obs_url\":"+"\""+output_dir+"\""+"}}}],\"engine\":{\"engine_name\":\"PyTorch\",\"engine_version\":\"pytorch_1.8.0-cuda_10.2-py_3.7-ubuntu_18.04-x86_64\"},\"local_code_dir\":\"/home/ma-user/modelarts/user-job-dir\",\"working_dir\":\"/home/ma-user/modelarts/user-job-dir\"},\"spec\":{\"resource\":{\"flavor_id\":\"modelarts.p3.large.public.free\",\"node_count\":1}}}"
    r.body = payload
    sig.Sign(r)
    resp = requests.request(r.method, r.scheme + "://" + r.host + r.uri, headers=r.headers, data=r.body)
    print(resp)
    resp_json = resp.json()
    job_id = resp_json["metadata"]["id"]
    status_code = resp.status_code
    return status_code,job_id

def get_mp4_num(log_dir: str, MP4_threshold: int) -> None:
    """Block until the recorded video count reaches the threshold.

    Polls a log file that records video filenames (one per line) and
    blocks the calling thread until the line count meets or exceeds
    ``MP4_threshold``, checking every 30 seconds.

    Args:
        log_dir: Path to the log file where video filenames are recorded.
        MP4_threshold: Minimum number of videos required to proceed
            with online training.
    """
    count = 0  # Fixed: was MP4_count, causing NameError since while loop used count
    while count < MP4_threshold:
        with open(log_dir) as f:
            count = len(f.readlines())
            print("The num of MP4 need to upload is" + str(count))
            time.sleep(30)

def ota() -> None:
    """Execute the full On-The-Air (OTA) model update pipeline.

    Orchestrates the end-to-end workflow for updating the edge device's
    YOLO model:

    1. Wait for enough recorded videos (``get_mp4_num``).
    2. Upload pre-trained weights and video dataset to OBS.
    3. Create and monitor a ModelArts fine-tuning job.
    4. Download the updated weights back to the edge device.
    5. Clean up the dataset directory on OBS.
    """
    # 读取文件
    get_mp4_num(log_dir, MP4_threshold)
    print("start online training")
    obs_client.putContent(bucket_name, upload_dir, '')
    print("start upload weight...")
    upload_file_to_obs(bucket_name,"./yolov5s.pt","yoloV5/yolov5/yolov5s.pt")
    print("start upload dataset...")
    upload_file_to_obs(bucket_name,"./videos",upload_dir[:-1])

    print("start creating train job...")
    status,job_id = create_training_job()
    if status != 201:
        print("create job error!")
        exit()
    print("Training job created, job_id: "+job_id)
    print("Training statue will be checked every 30 senconds...")
    while check_job_status(job_id)==False:
        time.sleep(30)
    #从OBS下载权重到本地
    print("Training job finished.")
    obs_client.getObject(bucket_name, "yoloV5/output/weights/best.pt", downloadPath='./yolov5s.pt')
    obs_client.getObject(bucket_name, "yoloV5/output/weights/best.onnx", downloadPath='./best.onnx')
    print("weight file downloaded.")

    #删除数据集目录
    resp = obs_client.listObjects(bucket_name, prefix=upload_dir)
    keys = []
    for content in resp.body.contents:
        keys.append(Object(key=content.key))
        # print('\t' + content.key + ' etag[' + content.etag + ']')
    # print(keys)
    resp = obs_client.deleteObjects(bucket_name, DeleteObjectsRequest(False, keys))

if __name__ == '__main__':
    ota()
