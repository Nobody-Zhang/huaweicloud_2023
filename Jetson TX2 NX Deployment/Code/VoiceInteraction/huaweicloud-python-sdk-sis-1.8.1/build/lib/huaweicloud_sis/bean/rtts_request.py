# -*- coding: utf-8 -*-


class RttsRequest:
    """ 实时语音转写请求，除了初始化必选参数外，其他参数均可不配置使用默认 """

    def __init__(self, text):
        """
            实时语音转写请求初始化
        :param audio_format:   音频格式，详见api文档
        :param model_property: 属性字符串，language_sampleRate_domain, 如chinese_8k_common，详见api文档
        """
        self._text = text
        self._command = 'START'
        self._audio_format = 'pcm'
        self._sample_rate = '8000'
        self._property = 'chinese_xiaoyan_common'
        self._speed = 0
        self._pitch = 0
        self._volume = 50

    def set_command(self, command):
        self._command = command

    def set_audio_format(self, audio_format):
        self._audio_format = audio_format

    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    def set_property(self, property):
        self._property = property

    def set_speed(self, speed):
        self._speed = speed

    def set_pitch(self, pitch):
        self._pitch = pitch

    def set_volume(self, volume):
        self._volume = volume

    def construct_params(self):
        config = dict()
        config['speed'] = self._speed
        config['pitch'] = self._pitch
        config['audio_format'] = self._audio_format
        config['sample_rate'] = self._sample_rate
        config['property'] = self._property
        config['volume'] = self._volume

        params = dict()
        params['command'] = self._command
        params['config'] = config
        params['text'] = self._text
        return params
