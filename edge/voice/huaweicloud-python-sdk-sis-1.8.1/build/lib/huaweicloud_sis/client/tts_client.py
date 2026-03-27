# -*- coding: utf-8 -*-

import json
from huaweicloud_sis.auth import aksk_service
from huaweicloud_sis.bean.tts_request import TtsCustomRequest
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.utils.logger_utils import logger


class TtsCustomizationClient:
    """ 语音合成client """
    def __init__(self, ak, sk, region, project_id, service_endpoint=None, sis_config=None):
        self._ak = ak
        self._sk = sk
        self._region = region
        self._project_id = project_id
        if service_endpoint is None:
            self._service_endpoint = 'https://sis-ext.' + region + '.myhuaweicloud.com'
        else:
            self._service_endpoint = service_endpoint
        if sis_config is None:
            self._sis_config = SisConfig()
        else:
            self._sis_config = sis_config

    def get_ttsc_response(self, request):
        """
            语音合成接口
        :param request: 语音合成请求，TtsCustomRequest
        :return: 请求结果，json格式
        """
        if not isinstance(request, TtsCustomRequest):
            logger.error('the parameter in \'get_ttsc_response(request)\' should be TtsCustomRequest class')
            raise ClientException('the parameter in \'get_ttsc_response(request)\' should be TtsCustomRequest class')
        url = self._service_endpoint + '/v1/' + self._project_id + '/tts'
        params = request.construct_params()
        headers = {'Content-Type': 'application/json'}
        result = aksk_service.aksk_connect(self._ak, self._sk, url, headers, params, 'POST', self._sis_config)
        if 'result' not in result:
            error_msg = 'The result of tts customization is invalid. Result is %s ' % json.dumps(result)
            logger.error(error_msg)
            raise ClientException(error_msg)
        if request.get_saved():
            base_str = result['result']['data']
            io_utils.save_audio_from_base64str(base_str, request.get_saved_path())
            result['is_saved'] = True
            result['saved_path'] = request.get_saved_path()
        return result
