# -*- coding: utf-8 -*-

from huaweicloud_sis.client.rasr_client import RasrClient
from huaweicloud_sis.bean.rasr_request import RasrRequest
from huaweicloud_sis.bean.callback import RasrCallBack
from huaweicloud_sis.bean.sis_config import SisConfig
import json

# 鉴权信息
ak = 'WAT9CMSRTF126WI93VAL'  # 用户的ak
sk = 'sADYWZRs46xg8ka6nynj7ul0Y9J4ky9T4bLSrAH7'  # 用户的sk
region = 'cn-north-4'  # region，如cn-north-4
project_id = '61ed29a1dbea43509f860d8e7ecb0942'  # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html

"""
    todo 请正确填写音频格式和模型属性字符串
    1. 音频格式一定要相匹配.
         例如音频是pcm格式，并且采样率为8k，则格式填写pcm8k16bit。
         如果返回audio_format is invalid 说明该文件格式不支持。具体支持哪些音频格式，需要参考一些api文档。

    2. 音频采样率要与属性字符串的采样率要匹配。
         例如格式选择pcm16k16bit，属性字符串却选择chinese_8k_common, 则会返回'audio_format' is not match model
"""

# 实时语音识别参数
path = ''  # 需要发送音频路径，如D:/test.pcm, 同时sdk也支持byte流发送数据。
audio_format = 'pcm16k16bit'  # 音频支持格式，如pcm16k16bit，详见api文档
property = 'chinese_16k_general'  # 属性字符串，language_sampleRate_domain, 如chinese_16k_general, 采样率要和音频一致。详见api文档


class MyCallback(RasrCallBack):
    """ 回调类，用户需要在对应方法中实现自己的逻辑，其中on_response必须重写 """

    def on_open(self):
        """ websocket连接成功会回调此函数 """
        print('websocket connect success')

    def on_start(self, message):
        """
            websocket 开始识别回调此函数
        :param message: 传入信息
        :return: -
        """
        print('webscoket start to recognize, %s' % message)

    def on_response(self, message):
        """
            websockert返回响应结果会回调此函数
        :param message: json格式
        :return: -
        """
        print(json.dumps(message, indent=2, ensure_ascii=False))

    def on_end(self, message):
        """
            websocket 结束识别回调此函数
        :param message: 传入信息
        :return: -
        """
        print('websocket is ended, %s' % message)

    def on_close(self):
        """ websocket关闭会回调此函数 """
        print('websocket is closed')

    def on_error(self, error):
        """
            websocket出错回调此函数
        :param error: 错误信息
        :return: -
        """
        print('websocket meets error, the error is %s' % error)

    def on_event(self, event):
        """
            出现事件的回调
        :param event: 事件名称
        :return: -
        """
        print('receive event %s' % event)


def rasr_example():
    """ 实时语音识别demo """
    # step1 初始化RasrClient, 暂不支持使用代理
    my_callback = MyCallback()
    config = SisConfig()
    # 设置连接超时,默认是10
    config.set_connect_timeout(10)
    # 设置读取超时, 默认是10
    config.set_read_timeout(10)
    # 设置connect lost超时，一般在普通并发下，不需要设置此值。默认是10
    config.set_connect_lost_timeout(10)
    # websocket暂时不支持使用代理
    rasr_client = RasrClient(ak=ak, sk=sk, use_aksk=True, region=region, project_id=project_id, callback=my_callback,
                             config=config)
    try:
        # step2 构造请求
        request = RasrRequest(audio_format, property)
        # 所有参数均可不设置，使用默认值
        request.set_add_punc('yes')  # 设置是否添加标点， yes or no， 默认no
        request.set_vad_head(10000)  # 设置有效头部， [0, 60000], 默认10000
        request.set_vad_tail(500)  # 设置有效尾部，[0, 3000]， 默认500
        request.set_max_seconds(30)  # 设置一句话最大长度，[0, 60], 默认30
        request.set_interim_results('no')  # 设置是否返回中间结果，yes or no，默认no
        request.set_digit_norm('no')  # 设置是否将语音中数字转写为阿拉伯数字，yes or no，默认yes
        # request.set_vocabulary_id('')     # 设置热词表id，若不存在则不填写，否则会报错
        request.set_need_word_info('no')  # 设置是否需要word_info，yes or no, 默认no

        # step3 选择连接模式
        # rasr_client.short_stream_connect(request)       # 流式一句话模式
        # rasr_client.sentence_stream_connect(request)    # 实时语音识别单句模式
        rasr_client.continue_stream_connect(request)  # 实时语音识别连续模式

        # step4 发送音频
        rasr_client.send_start()
        # 连续模式下，可多次发送音频，发送格式为byte数组
        with open(path, 'rb') as f:
            data = f.read()
            rasr_client.send_audio(data)  # 可选byte_len和sleep_time参数，建议使用默认值
        rasr_client.send_end()
    except Exception as e:
        print('rasr error', e)
    finally:
        # step5 关闭客户端，使用完毕后一定要关闭，否则服务端20s内没收到数据会报错并主动断开。
        rasr_client.close()


if __name__ == '__main__':
    rasr_example()
