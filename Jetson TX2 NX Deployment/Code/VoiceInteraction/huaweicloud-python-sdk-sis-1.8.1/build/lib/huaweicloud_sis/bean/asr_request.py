# -*- coding: utf-8 -*-
from huaweicloud_sis.bean.rasr_request import RasrRequest

class AsrCustomShortRequest:
    """ 一句话识别请求，除了初始化必选参数外，其他参数均可不配置使用默认 """
    def __init__(self, audio_format, model_property, data):
        """
            一句话识别请求初始化
        :param audio_format:    音频格式，详见api文档
        :param model_property:  language_sampleRate_domain, 如chinese_8k_common, 详见api文档
        :param data:            音频转化后的base64字符串
        """
        self._audio_format = audio_format
        self._property = model_property
        self._data = data
        self._add_punc = 'no'
        self._digit_norm = 'yes'
        self._vocabulary_id = None
        self._need_word_info = 'no'
        self._user_words = list()

    def set_add_punc(self, add_punc):
        self._add_punc = add_punc

    def set_digit_norm(self, digit_norm):
        self._digit_norm = digit_norm

    def set_vocabulary_id(self, vocabulary_id):
        self._vocabulary_id = vocabulary_id

    def set_need_word_info(self, need_word_info):
        self._need_word_info = need_word_info

    def set_user_words(self, user_words):
        self._user_words = user_words

    def construct_params(self):
        params = dict()
        params['data'] = self._data
        config = dict()
        config['audio_format'] = self._audio_format
        config['property'] = self._property
        config['add_punc'] = self._add_punc
        config['digit_norm'] = self._digit_norm
        config['need_word_info'] = self._need_word_info
        if self._vocabulary_id is not None:
            config['vocabulary_id'] = self._vocabulary_id
        if self._user_words is not None and len(self._user_words) > 0:
            config['user_words'] = self._user_words
        params['config'] = config
        return params


class AsrCustomLongRequest:
    """ 录音文件识别请求，除了初始化必选参数外，其他参数均可不配置使用默认 """
    def __init__(self, audio_format, model_property, data_url):
        """
            录音文件识别初始化
        :param audio_format:   音频格式，详见api文档
        :param model_property: 属性字符串，language_sampleRate_domain, 详见api文档
        :param data_url:       音频的obs链接
        """
        self._audio_format = audio_format
        self._property = model_property
        self._data_url = data_url
        self._add_punc = 'no'
        self._digit_norm = 'yes'
        self._callback_url = None
        self._need_analysis_info = False
        self._diarization = True
        self._channel = 'MONO'
        self._emotion = True
        self._speed = True
        self._vocabulary_id = None
        self._need_word_info = 'no'

    def set_callback_url(self, callback_url):
        self._callback_url = callback_url

    def set_add_punc(self, add_punc):
        self._add_punc = add_punc

    def set_digit_norm(self, digit_norm):
        self._digit_norm = digit_norm

    def set_need_analysis_info(self, need_analysis_info):
        self._need_analysis_info = need_analysis_info

    def set_diarization(self, diarization):
        self._diarization = diarization

    def set_channel(self, channel):
        self._channel = channel

    def set_emotion(self, emotion):
        self._emotion = emotion

    def set_speed(self, speed):
        self._speed = speed

    def set_vocabulary_id(self, vocabulary_id):
        self._vocabulary_id = vocabulary_id

    def set_need_word_info(self, need_word_info):
        self._need_word_info = need_word_info

    def construct_parameter(self):
        params = dict()
        params['data_url'] = self._data_url
        config = dict()
        config['audio_format'] = self._audio_format
        config['property'] = self._property
        config['add_punc'] = self._add_punc
        config['digit_norm'] = self._digit_norm
        config['need_word_info'] = self._need_word_info
        if self._callback_url is not None and not self._callback_url == '':
            config['callback_url'] = self._callback_url
        if self._need_analysis_info:
            need_analysis_info = dict()
            need_analysis_info['diarization'] = self._diarization
            need_analysis_info['channel'] = self._channel
            need_analysis_info['emotion'] = self._emotion
            need_analysis_info['speed'] = self._speed
            config['need_analysis_info'] = need_analysis_info
        if self._vocabulary_id is not None:
            config['vocabulary_id'] = self._vocabulary_id
        params['config'] = config
        return params


class SasrWebsocketRequest(RasrRequest):
    def __init__(self, audio_format, model_property):
        super().__init__(audio_format, model_property)

    def construct_params(self):
        result = super().construct_params()
        remove_list = ['vad_head', 'vad_tail', 'max_seconds']
        for key in remove_list:
            if key in result:
                del result[key]
        return result

