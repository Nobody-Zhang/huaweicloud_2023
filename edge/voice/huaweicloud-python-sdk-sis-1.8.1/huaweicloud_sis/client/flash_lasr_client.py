# -*- coding: utf-8 -*-

from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.bean.flash_lasr_request import FlashLasrRequest
from huaweicloud_sis.utils.logger_utils import logger
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.auth import aksk_service


class FlashLasrClient:
    """ 录音文件极速版 client """

    def __init__(self, ak, sk, region, project_id, service_endpoint=None, sis_config=None):
        """
            录音文件极速版client初始化
        :param ak:                  ak
        :param sk:                  sk
        :param region:              区域，如cn-north-4
        :param project_id:          项目id，可参考https://support.huaweicloud.com/api-sis/sis_03_0008.html
        :param service_endpoint:    终端节点，可不填使用默认即可
        :param sis_config:          配置信息，包括超时、代理等，可不填使用默认即可。
        """
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

    def get_flash_lasr_result(self, request):
        """
            录音文件极速版接口
        :param request: 录音文件极速版请求
        :return: 响应结果，返回为json格式
        """
        if not isinstance(request, FlashLasrRequest):
            error_msg = 'the parameter in flash lasr should be FlashLasrRequest class'
            logger.error(error_msg)
            raise ClientException(error_msg)
        url = self._service_endpoint + '/v1/' + self._project_id + '/asr/flash'
        query_url = url + request.construct_params()
        headers = {'Content-Type': 'application/json'}
        result = aksk_service.aksk_connect(self._ak, self._sk, query_url, headers, None, 'POST', self._sis_config)
        return result