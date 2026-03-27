# -*- coding: utf-8 -*-

from huaweicloud_sis.client.pa_client import PaClient
from huaweicloud_sis.bean.pa_request import PaAudioRequest
from huaweicloud_sis.bean.pa_request import PaVideoRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.sis_config import SisConfig
import json

# 鉴权参数
ak = ''  # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
sk = ''  # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
region = ''  # region，目前仅支持cn-north-4，参考https://support.huaweicloud.com/api-sis/sis_03_0004.html
project_id = ''  # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html

# 语音评测参数, 根据语音和文本对发音进行评分。
audio_path = ''             # 音频文件位置, 如D:/test.wav, sdk支持将文件转化为base64编码
audio_ref_text = ''         # 音频对应文本，用于评分

# 多模态评测参数，根据视频和语音及文本，对发音进行评分、
video_path = ''             # 视频文件位置, 如D:/test.mp4, sdk支持将文件转化为base64编码
video_ref_text = ''         # 视频中音频对应文本，用于评分


def assessment_audio_example():
    """ 语音评测示例 """
    # step1 初始化客户端
    config = SisConfig()
    config.set_connect_timeout(10)  # 设置连接超时
    config.set_read_timeout(10)     # 设置读取超时
    # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
    # config.set_proxy(proxy)
    pa_client = PaClient(ak, sk, region, project_id, sis_config=config)

    # step2 构造请求
    audio_data = io_utils.encode_file(audio_path)
    pa_audio_request = PaAudioRequest()
    # 设置音频的base64编码
    pa_audio_request.set_audio_data(audio_data)
    # 设置音频的标准文本
    pa_audio_request.set_ref_text(audio_ref_text)
    # 设置音频格式，具体支持格式详见api文档。
    pa_audio_request.set_audio_format('auto')
    # 设置语音，默认en_gb，具体支持详见api文档
    pa_audio_request.set_language('en_gb')
    # 设置模式，word 或 sentence，默认word，具体支持详见api文档。
    pa_audio_request.set_mode('word')

    # step3 发送请求，返回结果,返回结果为json格式
    result = pa_client.assessment_audio(pa_audio_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def assessment_vidoe_example():
    """ 多模态评测示例 """
    # step1 初始化客户端
    config = SisConfig()
    config.set_connect_timeout(5)  # 设置连接超时
    config.set_read_timeout(10)  # 设置读取超时
    # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
    # config.set_proxy(proxy)
    pa_client = PaClient(ak, sk, region, project_id, sis_config=config)

    # step2 构造请求
    video_data = io_utils.encode_file(video_path)
    pa_video_request = PaVideoRequest()
    # 设置视频的base64编码
    pa_video_request.set_video_data(video_data)
    # 设置视频的标准文本
    pa_video_request.set_ref_text(video_ref_text)
    # 设置视频格式，具体支持格式详见api文档。
    pa_video_request.set_video_format('auto')
    # 设置语音，默认en_gb，具体支持详见api文档
    pa_video_request.set_language('en_gb')
    # 设置模式，word 或 sentence，默认word，具体支持详见api文档。
    pa_video_request.set_mode('word')

    # step3 发送请求，返回结果,返回结果为json格式
    result = pa_client.assessment_video(pa_video_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        assessment_audio_example()     # 语音评测
        assessment_vidoe_example()     # 多模态评测
    except ClientException as e:
        print(e)
    except ServerException as e:
        print(e)