# -*- coding: utf-8 -*-

from huaweicloud_sis.utils.logger_utils import logger
from huaweicloud_sis.exception.exceptions import ClientException


class PaAudioRequest:
    """ 语音评测请求 """
    def __init__(self):
        """
            语音请求初始化
        """
        self._audio_data = None
        self._audio_url = None
        self._ref_text = None
        self._audio_format = 'auto'
        self._language = 'en_gb'
        self._mode = 'word'

    def set_audio_data(self, audio_data):
        self._audio_data = audio_data

    def set_audio_url(self, audio_url):
        self._audio_url = audio_url

    def set_audio_format(self, audio_format):
        self._audio_format = audio_format

    def set_ref_text(self, ref_text):
        self._ref_text = ref_text

    def set_language(self, language):
        self._language = language

    def set_mode(self, mode):
        self._mode = mode

    def construct_params(self):
        if self._audio_data is None and self._audio_url is None:
            logger.error('In PaAudioRequest, audio_data and audio_url can\'t be both empty')
            raise ClientException('In PaAudioRequest, audio_data and audio_url can\'t be both empty')
        if self._ref_text is None:
            logger.error('In PaAudioRequest, ref_text can\'t be empty')
            raise ClientException('In PaAudioRequest, ref_text can\'t be empty')
        if self._audio_data is not None and self._audio_url is not None:
            logger.warn('When audio_data and audio_url are all filled, only audio_data takes effect')
        params_dict = {
            'ref_text': self._ref_text,
            'config': {
                'audio_format': self._audio_format,
                'language': self._language,
                'mode': self._mode
            }
        }
        if self._audio_data is not None:
            params_dict['audio_data'] = self._audio_data
        else:
            params_dict['audio_url'] = self._audio_url
        return params_dict


class PaVideoRequest:
    def __init__(self):
        """
            多模态评测请求

        """
        self._video_data = None
        self._video_url = None
        self._video_format = 'auto'
        self._ref_text = None
        self._language = 'en_gb'
        self._mode = 'word'

    def set_video_data(self, video_data):
        self._video_data = video_data

    def set_video_url(self, video_url):
        self._video_url = video_url

    def set_video_format(self, video_format):
        self._video_format = video_format

    def set_ref_text(self, ref_text):
        self._ref_text = ref_text

    def set_language(self, language):
        self._language = language

    def set_mode(self, mode):
        self._mode = mode

    def construct_parameter(self):
        if self._video_data is None and self._video_url is None:
            logger.error('In PaVideoRequest, video_data and video_url can\'t be both empty')
            raise ClientException('In PaVideoRequest, video_data and video_url can\'t be both empty')
        if self._ref_text is None:
            logger.error('In PaVideoRequest, ref_text can\'t be empty')
            raise ClientException('In PaVideoRequest, ref_text can\'t be empty')
        if self._video_data is not None and self._video_url is not None:
            logger.warn('When video_data and video_url are all filled, only video_data takes effect')
        params_dict = {
            'ref_text': self._ref_text,
            'config': {
                'video_format': self._video_format,
                'language': self._language,
                'mode': self._mode
            }
        }
        if self._video_data is not None:
            params_dict['video_data'] = self._video_data
        else:
            params_dict['video_url'] = self._video_url
        return params_dict
