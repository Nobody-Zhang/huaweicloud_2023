import requests
import json

def get_huawei_cloud_token():
    # 华为云认证服务的请求URL
    auth_url = "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens"

    # 提供身份验证信息
    auth_payload = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "name": "test",
                        "password": "qwer1234",
                        "domain": {
                            "name": "nobody_zgb"
                        }
                    }
                }
            },
            "scope": {
                "project": {
                    "name": "cn-north-4"
                }
            }
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    # 发送认证请求
    response = requests.post(auth_url, headers=headers, data=json.dumps(auth_payload))
    print(response)

    # 检查响应状态码
    if response.status_code == 201:
        # 提取令牌信息
        token = response.headers[('X-Subject-Token')]
        return token
    else:
        print("认证失败，HTTP 状态码:", response.status_code)
        return None

if __name__ == '__main__':
    token = get_huawei_cloud_token()
    if token:
        print("获取到的令牌:", token)
    else:
        print("无法获取令牌。")

