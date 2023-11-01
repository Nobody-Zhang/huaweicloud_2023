# -*- coding: utf-8 -*-

from huaweicloud_sis.client.tts_client import TtsCustomizationClient
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
import json


def ttsc_example():
    """ 语音合成demo """
    ak = 'WAT9CMSRTF126WI93VAL'             # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
    sk = 'sADYWZRs46xg8ka6nynj7ul0Y9J4ky9T4bLSrAH7'             # 参考https://support.huaweicloud.com/sdkreference-sis/sis_05_0003.html
    region = 'cn-north-4'         # region，如cn-north-4
    project_id = '61ed29a1dbea43509f860d8e7ecb0942'     # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html
    text = ''           # 待合成文本，不超过500字
    path = ''           # 保存路径，如D:/test.wav。 可在设置中选择不保存本地

    # step1 初始化客户端
    config = SisConfig()
    config.set_connect_timeout(10)       # 设置连接超时，单位s
    config.set_read_timeout(10)          # 设置读取超时，单位s
    # 设置代理，使用代理前一定要确保代理可用。 代理格式可为[host, port] 或 [host, port, username, password]
    # config.set_proxy(proxy)
    ttsc_client = TtsCustomizationClient(ak, sk, region, project_id, sis_config=config)

    # step2 构造请求
    ttsc_request = TtsCustomRequest(text)
    # 设置请求，所有参数均可不设置，使用默认参数
    # 设置属性字符串， language_speaker_domain, 默认chinese_xiaoyan_common, 参考api文档
    ttsc_request.set_property('chinese_xiaoyan_common')
    # 设置音频格式，默认wav，可选mp3和pcm
    ttsc_request.set_audio_format('wav')
    # 设置采样率，8000 or 16000, 默认8000
    ttsc_request.set_sample_rate('8000')
    # 设置音量，[0, 100]，默认50
    ttsc_request.set_volume(50)
    # 设置音高, [-500, 500], 默认0
    ttsc_request.set_pitch(0)
    # 设置音速, [-500, 500], 默认0
    ttsc_request.set_speed(0)
    # 设置是否保存，默认False
    ttsc_request.set_saved(True)
    # 设置保存路径，只有设置保存，此参数才生效
    ttsc_request.set_saved_path(path)

    # step3 发送请求，返回结果。如果设置保存，可在指定路径里查看保存的音频。
    result = ttsc_client.get_ttsc_response(ttsc_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        ttsc_example()
    except ClientException as e:
        print(e)
    except ServerException as e:
        print(e)
