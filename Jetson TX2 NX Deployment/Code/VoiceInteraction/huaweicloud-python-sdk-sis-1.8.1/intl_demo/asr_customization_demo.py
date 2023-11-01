# -*- coding: utf-8 -*-

from huaweicloud_sis.client.asr_client import AsrCustomizationClient
from huaweicloud_sis.bean.asr_request import AsrCustomShortRequest
from huaweicloud_sis.exception.exceptions import ClientException
from huaweicloud_sis.exception.exceptions import ServerException
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.sis_config import SisConfig
import json

# personal information
ak = ''
sk = ''
region = ''         # region, such as ap-southeast-3
project_id = ''     # project_id, refer to https://support.huaweicloud.com/intl/en-us/api-sis/sis_03_0008.html

path = ''                   # the file path，such as D:/test.wav. The sample rate should be same as pathProperty.
path_audio_format = ''      # audio format，such as wav. Please refer to api document
path_property = ''          # language_sampleRate_domain, such as english_8k_common. Please refer to api document


def asrc_short_example():
    """ Sentence Transcription demo """
    # step1 initialize client
    config = SisConfig()
    config.set_connect_timeout(10)
    config.set_read_timeout(10)
    # set proxy, please make sure that it can take effects before use.
    # proxy = [host, port] or proxy = [host, port, username, password]
    # config.set_proxy(proxy)
    asr_client = AsrCustomizationClient(ak, sk, region, project_id,  sis_config=config)

    # step2 set the parameters
    base64_str = io_utils.encode_file(path)    # the audio file will be converted to base64 str
    asr_request = AsrCustomShortRequest(path_audio_format, path_property, base64_str)
    # all parameters are optional.
    # set punctuation, yes or no, default no.
    asr_request.set_add_punc('yes')
	# sentence transcription doesn't support hot word now. So the method of set_vocabulary_id makes no sense. 

    # step3 send request, get response
    result = asr_client.get_short_response(asr_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        asrc_short_example()
    except ClientException as e:
        print(e)
    except ServerException as e:
        print(e)