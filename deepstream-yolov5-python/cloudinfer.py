import requests
from apig_sdk import signer

def cloud_infer(file_path = "tmp.mp4"):
    # Config url, ak, sk and file path.
    url = "https://5057cfd5b15e4d70abd1bfdf3ae0c128.apig.cn-north-4.huaweicloudapis.com/v1/infers/9fedb564-8930-4bc1-b821-a355fae3538c"
    ak = "ISZL2F2AUMZO1LPRMIJK"
    sk = "7wqL0eQl8ZduQWDQr3NjuXtDPLBIikgjxDvx3uaa"

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
    print(resp.status_code)
    print(resp.text)

    return resp