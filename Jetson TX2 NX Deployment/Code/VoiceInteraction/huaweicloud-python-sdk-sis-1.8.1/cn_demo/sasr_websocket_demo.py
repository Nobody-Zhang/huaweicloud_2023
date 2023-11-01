# -*- coding: utf-8 -*-

from huaweicloud_sis.client.asr_client import SasrWebsocketClient
from huaweicloud_sis.bean.asr_request import SasrWebsocketRequest
from huaweicloud_sis.bean.callback import RasrCallBack
from huaweicloud_sis.bean.sis_config import SisConfig
import json

# 鉴权信息
ak = ''  # 用户的ak
sk = ''  # 用户的sk
region = 'cn-north-4'  # region，如cn-north-4
project_id = ''  # 同region一一对应，参考https://support.huaweicloud.com/api-sis/sis_03_0008.html

# 一句话识别参数
path = ''           # 需要发送音频路径，如D:/test.pcm, 同时sdk也支持byte流发送数据。
audio_format = ''   # 音频支持格式，如pcm16k16bit，详见api文档
property = ''       # 属性字符串，language_sampleRate_domain, 如chinese_16k_common, 采样率要和音频一致。详见api文档


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


def sasr_websocket_example():
    """ 一句话识别 websocket demo """
    # step1 初始化SasrWebsocketClient, 暂不支持使用代理
    my_callback = MyCallback()
    config = SisConfig()
    # 设置连接超时,默认是10
    config.set_connect_timeout(10)
    # 设置读取超时, 默认是10
    config.set_read_timeout(10)
    # 设置connect lost超时，一般在普通并发下，不需要设置此值。默认是10
    config.set_connect_lost_timeout(10)
    # websocket暂时不支持使用代理
    sasr_websocket_client = SasrWebsocketClient(ak=ak, sk=sk, use_aksk=True, region=region, project_id=project_id,
                                                callback=my_callback, config=config)
    try:
        # step2 构造请求
        request = SasrWebsocketRequest(audio_format, property)
        # 所有参数均可不设置，使用默认值
        request.set_add_punc('yes')  # 设置是否添加标点， yes or no， 默认no
        request.set_interim_results('no')  # 设置是否返回中间结果，yes or no，默认no
        request.set_digit_norm('no')  # 设置是否将语音中数字转写为阿拉伯数字，yes or no，默认yes
        # request.set_vocabulary_id('')     # 设置热词表id，若不存在则不填写，否则会报错
        request.set_need_word_info('no')  # 设置是否需要word_info，yes or no, 默认no

        # step3 连接服务端
        sasr_websocket_client.sasr_stream_connect(request)

        # step4 发送音频
        sasr_websocket_client.send_start()
        # 连续模式下，可多次发送音频，发送格式为byte数组
        with open(path, 'rb') as f:
            data = f.read()
            sasr_websocket_client.send_audio(data)  # 可选byte_len和sleep_time参数，建议使用默认值
        sasr_websocket_client.send_end()
    except Exception as e:
        print('sasr websocket error', e)
    finally:
        # step5 关闭客户端，使用完毕后一定要关闭，否则服务端20s内没收到数据会报错并主动断开。
        sasr_websocket_client.close()


if __name__ == '__main__':
    sasr_websocket_example()
