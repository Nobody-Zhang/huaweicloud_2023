import os
import requests
from apig_sdk import signer

import logging
logger = logging.getLogger(__name__)
def cloud_infer(file_path = "tmp.mp4"):
    # Config url, ak, sk and file path.
    url = os.environ.get("HUAWEICLOUD_CLOUDINFER_URL", "")
    ak = os.environ.get("HUAWEICLOUD_AK")
    sk = os.environ.get("HUAWEICLOUD_SK")

    # Create request, set method, url, headers and body.
    method = 'POST'
    headers = {"x-sdk-content-sha256": "UNSIGNED-PAYLOAD"}
    request = signer.HttpRequest(method, url, headers)

    # Create sign, set the AK/SK to sign and authenticate the request.
    sig = signer.Signer()
    sig.Key = ak
    sig.Secret = sk
    sig.Sign(request)

    # Send request
    files = {'file': open(file_path, 'rb')}
    resp = requests.request(request.method, request.scheme + "://" + request.host + request.uri, headers=request.headers, files=files)

    # # Print result
    logger.info(resp.status_code)
    logger.info(resp.text)

    return resp