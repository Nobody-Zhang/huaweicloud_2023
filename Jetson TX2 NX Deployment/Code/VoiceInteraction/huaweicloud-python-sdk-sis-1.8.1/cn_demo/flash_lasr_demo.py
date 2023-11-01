# -*- coding: utf-8 -*-

from huaweicloud_sis.client.flash_lasr_client import FlashLasrClient
from huaweicloud_sis.bean.flash_lasr_request import FlashLasrRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.bean.sis_config import SisConfig
import json

# 鉴权参数
ak = ''             # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
sk = ''             # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
region = ''         # region，如cn-north-4
project_id = ''     # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html

obs_bucket_name = ''    # obs桶名
obs_object_key = ''     # obs对象的key
property = ''           # 文件格式，如wav等， 支持格式详见api文档
audio_format = ''       # 属性字符串，language_sampleRate_domain, 如chinese_8k_common, 详见api文档


def flash_lasr_example():
    """ 录音文件极速版示例 """
    # step1 初始化客户端
    config = SisConfig()
    config.set_connect_timeout(10)  # 设置连接超时
    config.set_read_timeout(10)  # 设置读取超时
    # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
    # config.set_proxy(proxy)
    client = FlashLasrClient(ak, sk, region, project_id, sis_config=config)

    # step2 构造请求
    asr_request = FlashLasrRequest()
    # 以下参数必选
    # 设置存放音频的桶名，必选
    asr_request.set_obs_bucket_name(obs_bucket_name)
    # 设置桶内音频对象名，必选
    asr_request.set_obs_object_key(obs_object_key)
    # 设置格式，必选
    asr_request.set_audio_format(audio_format)
    # 设置属性，必选
    asr_request.set_property(property)

    # 以下参数可选
    # 设置是否添加标点，yes or no，默认no
    asr_request.set_add_punc('yes')
    # 设置是否将语音中数字转写为阿拉伯数字，yes or no，默认yes
    asr_request.set_digit_norm('yes')
    # 设置是否添加热词表id，没有则不填
    # asr_request.set_vocabulary_id(None)
    # 设置是否需要word_info，yes or no, 默认no
    asr_request.set_need_word_info('no')
    # 设置是否只识别收个声道的音频数据，默认no
    asr_request.set_first_channel_only('no')

    # step3 发送请求，返回结果,返回结果为json格式
    result = client.get_flash_lasr_result(asr_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        flash_lasr_example()
    except ClientException as e:
        print(e)
    except ServerException as e:
        print(e)
