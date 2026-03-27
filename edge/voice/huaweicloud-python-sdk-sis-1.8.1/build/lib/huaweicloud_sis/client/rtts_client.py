# -*- coding: utf-8 -*-

import websocket
from huaweicloud_sis.utils.logger_utils import logger
from huaweicloud_sis.auth import token_service
from huaweicloud_sis.exception.exceptions import ClientException, ServerException
from huaweicloud_sis.bean.sis_enum import WebsocketStatus
from huaweicloud_sis.bean.callback import RttsCallBack
from huaweicloud_sis.bean.sis_config import SisConfig
import ssl
from huaweicloud_sis.bean.rtts_request import RttsRequest
from huaweicloud_sis.auth import aksk_service
import json
import time
import threading

user_dict = dict()
time_dict = dict()

connect_sleep_time = 0.01


class RttsClient:
    """ 实时语音合成client """

    def __init__(self, user_name=None, password=None, domain_name=None, region=None, project_id=None, callback=None,
                 config=SisConfig(), service_endpoint=None, token_url=None, retry_sleep_time=1,
                 ak=None, sk=None, use_aksk=False):
        """
            实时语音合成client初始化。 推荐使用ak、sk认证方式。
        :param user_name:           用户名，当use_aksk为false生效.
        :param password:            密码，当use_aksk为false生效
        :param domain_name:         账户名，一般等同用户名，当use_aksk为false生效
        :param region:              区域，如cn-north-4
        :param project_id:          项目ID，可参考https://support.huaweicloud.com/api-sis/sis_03_0008.html
        :param callback:            回调类RttsCallBack，用于监听websocket连接、响应、断开、错误等
        :param service_endpoint:    终端节点，一般使用默认即可
        :param token_url:           请求token的url，一般使用默认即可
        :param retry_sleep_time:    当websocket连接失败重试的间隔时间，默认为1s
        :param ak                   ak, 当 use_aksk为true生效。
        :param sk                   sk, 当 use_aksk为true生效。
        :param use_aksk             是否使用ak sk认证，为true表示选择ak sk认证。为false表示使用token认证。
        """
        if service_endpoint is None:
            self._service_endpoint = 'wss://sis-ext.' + region + '.myhuaweicloud.com'
        else:
            self._service_endpoint = service_endpoint
        if not isinstance(callback, RttsCallBack):
            logger.error('The parameter callback must be RttsCallBack class')
            raise ClientException('The parameter callback must be RttsCallBack class')
        if not isinstance(config, SisConfig):
            logger.error('The parameter config must by SisConfig class')
            raise ClientException('The parameter config must by SisConfig class')

        self._region = region
        self._project_id = project_id
        self._config = config
        self._callback = callback
        self._status = WebsocketStatus.STATE_INIT
        self._retry_sleep_time = retry_sleep_time
        self._use_aksk = use_aksk
        self._thread = None
        self._ws = None
        if use_aksk:
            if ak is None or sk is None:
                logger.error('you choose aksk authentication, ak or sk is empty, please fill in it')
                raise ClientException('ak or sk is empty, please fill in it')
            self._ak = ak
            self._sk = sk
        else:
            if user_name is None or password is None or domain_name is None:
                logger.error('you choose token authentication, username or password or domain name is empty, '
                             'please fill in it')
                raise ClientException('username or password or domain name is empty, please fill in it')
            self._token = self._get_token(user_name, password, domain_name, token_url)

        # 此变量正常情况下永远为False。除非依赖库websocket-cliet变更，不再自动添加header
        # 1. websocket的库会在请求头部自动添加host信息。而且websocket库的header格式为tuple，即 ['key1:val1', 'key2:val2']
        # 2. apig加密的header也包含host信息。
        # 3. 当将apig加密后的header放在websocket请求中，会形成两个host信息，且无法覆盖。导致无法通过apig鉴权，报错401。
        # 4. 将其设置为false，可以屏蔽apig加密header里的host信息，最终形成header只有一个host，可通过鉴权。
        self._show_host_header = False

    def _connect(self, url):
        def _on_open(ws):
            logger.info('rtts websocket open')
            self._status = WebsocketStatus.STATE_CONNECTED
            self._callback.on_open()

        def _on_message(ws, message):
            if isinstance(message, bytes):
                self._callback.on_response(data=message)
            elif isinstance(message, str):
                result = json.loads(message)
                result_type = result['resp_type']
                trace_id = result['trace_id']
                if result_type == 'START':
                    self._status = WebsocketStatus.STATE_START
                    self._callback.on_start('trace id is %s' % trace_id)
                elif result_type == 'END':
                    self._status = WebsocketStatus.STATE_END
                    self._callback.on_end('trace id is %s' % trace_id)
                elif result_type == 'ERROR' or result_type == 'FATAL_ERROR':
                    logger.error("receive error resp %s" % message)
                    self._status = WebsocketStatus.STATE_ERROR
                    self._callback.on_error(message)
                else:
                    self._status = WebsocketStatus.STATE_ERROR
                    logger.error('%s don\'t belong to any type' % result_type)
            else:
                logger.error('don\'t support message type %s' % str(message))
                self._status = WebsocketStatus.STATE_ERROR

        def _on_close(ws, close_status_code='1000', close_msg='close'):
            logger.info('rtts websocket close')
            self._status = WebsocketStatus.STATE_CLOSE
            self._callback.on_close()

        def _on_error(ws, error):
            logger.error(error)
            self._status = WebsocketStatus.STATE_ERROR
            self._callback.on_error(error)

        if self._config.get_certificate_check():
            sslopt = None
        else:
            sslopt = {"cert_reqs": ssl.CERT_NONE}

        if not self._check_connect():
            logger.warning("status %s can't connect rtts,it will not execute connect" % self._status)
            return
        headers = self._get_headers(url)
        retry_count = 3
        self._status = WebsocketStatus.STATE_CONNECT_WAITING
        for i in range(retry_count):
            self._ws = websocket.WebSocketApp(url, headers, on_open=_on_open, on_close=_on_close,
                                              on_message=_on_message, on_error=_on_error)

            self._thread = threading.Thread(target=self._ws.run_forever,
                                            args=(None, sslopt, self._config.get_connect_lost_timeout() + 10,
                                                  self._config.get_connect_lost_timeout()))
            self._thread.daemon = True
            self._thread.start()

            try:
                status_set = {WebsocketStatus.STATE_CONNECTED, WebsocketStatus.STATE_CLOSE, WebsocketStatus.STATE_ERROR}
                self._wait_status(status_set, self._config.get_connect_timeout())
                break
            except Exception as e:
                logger.warning("connect occurs exception %s" % str(e))
        if self._status != WebsocketStatus.STATE_CONNECTED:
            error_msg = 'websocket connect failed， url is %s, status is %s' % (url, self._status)
            logger.error(error_msg)
            raise ClientException(error_msg)

    def synthesis(self, request):
        try:
            self._connect_to_websocket()
            self._send_request(request)
        finally:
            self._close()

    def _send_request(self, request):
        """ 发送合成请求 """
        if self._check_request(request):
            self._status = WebsocketStatus.STATE_END_WAITING
            message = json.dumps(request.construct_params())
            self._ws.send(message, opcode=websocket.ABNF.OPCODE_TEXT)
            wait_set = {WebsocketStatus.STATE_END, WebsocketStatus.STATE_CLOSE, WebsocketStatus.STATE_ERROR}
            self._wait_status(wait_set, self._config.get_websocket_wait_time())

    def _close(self):
        """ 服务超过20s没收到请求会自动断开 """
        if self._thread and self._thread.is_alive():
            self._ws.keep_running = False
            self._thread.join()
        if self._ws is not None:
            self._ws.close()

    def _connect_to_websocket(self):
        if self._check_connect():
            url = self._service_endpoint + '/v1/' + self._project_id + '/rtts'
            self._connect(url)

    def _set_host_header(self, show_host_header):
        """
            正常可访问情况下，永远不要调用这个接口。默认屏蔽，除非依赖库websocket-client变更，不再自动添加header中的host信息。
        :return:
        """
        self._show_host_header = show_host_header

    def _check_request(self, request):
        if not isinstance(request, RttsRequest):
            error_msg = 'The parameter of request in RttsClient should be RttsRequest class'
            logger.error(error_msg)
            raise ClientException(error_msg)
        if self._status == WebsocketStatus.STATE_CONNECTED:
            return True
        logger.error("status {} can't send request", self._status)
        return False

    def _get_headers(self, url):
        if self._use_aksk:
            if not url.startswith('ws'):
                logger.error('%s is invalid' % url)
                raise ClientException('%s is invalid' % url)
            new_url = 'http' + url[2:]
            headers = aksk_service.get_signed_headers(self._ak, self._sk, new_url, None, None, 'GET')
            if 'host' in headers and not self._show_host_header:
                del headers['host']
        else:
            headers = {'X-Auth-Token': self._token}
        return headers

    def _get_token(self, user_name, password, domain_name, token_url):
        if self._use_aksk:
            return None
        if token_url is None:
            token_url = 'https://iam.' + self._region + '.myhuaweicloud.com/v3/auth/tokens'
        # token 缓存必须在client进行，才可以在多线程中生效。
        now_time = time.time()
        token = None
        if user_name in user_dict and user_name in time_dict:
            temp_token = user_dict[user_name]
            save_time = time_dict[user_name]
            if now_time - save_time < 5 * 3600:
                logger.info('use token cache')
                token = temp_token
        if token is None:
            token = token_service.get_token(user_name, password, domain_name, self._region,
                                            url=token_url, config=self._config)
            user_dict[user_name] = token
            time_dict[user_name] = now_time
        return token

    def _wait_status(self, target_status_set, max_wait_time, sleep_time=0.01):
        wait_count = int(max_wait_time / sleep_time)
        for i in range(wait_count):
            if self._status in target_status_set:
                return
            time.sleep(sleep_time)
        error_msg = "status %s is not received %s in %s time" % (self._status, target_status_set, max_wait_time)
        logger.error(error_msg)
        raise ClientException(error_msg)

    def _check_connect(self):
        if self._status == WebsocketStatus.STATE_ERROR:
            self._close()
            return True
        if self._status == WebsocketStatus.STATE_INIT or self._status == WebsocketStatus.STATE_CLOSE:
            return True
        logger.warning("status %s can't connect" % self._status)
        return False

    def _check_send(self):
        if self._status == WebsocketStatus.STATE_CONNECTED:
            return True
        error_msg = "status %s can't send command" % self._status
        logger.error(error_msg)
        raise ClientException(error_msg)
