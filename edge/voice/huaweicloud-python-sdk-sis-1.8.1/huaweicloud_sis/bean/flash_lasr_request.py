# -*- coding: utf-8 -*-

class FlashLasrRequest:
    def __init__(self):
        self._audio_format = None
        self._property = None
        self._add_punc = 'no'
        self._digit_norm = 'no'
        self._vocabulary_id = None
        self._need_word_info = 'no'
        self._first_channel_only = 'no'
        self._obs_bucket_name = None
        self._obs_object_key = None

    def set_audio_format(self, audio_format):
        self._audio_format = audio_format

    def set_property(self, property):
        self._property = property

    def set_add_punc(self, add_punc):
        self._add_punc = add_punc

    def set_digit_norm(self, digit_norm):
        self._digit_norm = digit_norm

    def set_vocabulary_id(self, vocabulary_id):
        self._vocabulary_id = vocabulary_id

    def set_need_word_info(self, need_word_info):
        self._need_word_info = need_word_info

    def set_first_channel_only(self, first_channel_only):
        self._first_channel_only = first_channel_only

    def set_obs_bucket_name(self, obs_bucket_name):
        self._obs_bucket_name = obs_bucket_name

    def set_obs_object_key(self, obs_object_key):
        self._obs_object_key = obs_object_key

    def construct_params(self):
        params = '?'
        if self._audio_format is not None:
            params += '&audio_format=' + str(self._audio_format)
        if self._property is not None:
            params += '&property=' + str(self._property)
        params += '&add_punc=' + str(self._add_punc)
        params += '&digit_norm=' + str(self._digit_norm)

        if self._vocabulary_id is not None:
            params += '&vocabulary_id=' + str(self._vocabulary_id)
        params += '&need_word_info=' + str(self._need_word_info)
        params += '&first_channel_only=' + str(self._first_channel_only)
        if self._obs_bucket_name is not None:
            params += '&obs_bucket_name=' + str(self._obs_bucket_name)
        if self._obs_object_key is not None:
            params += '&obs_object_key=' + str(self._obs_object_key)
        return params

