/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "gst-nvdssr.h"
#include "nvdsinfer.h"

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <cmath>
#include <tuple>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <curl/curl.h>
#include <json/json.h>

#include "deepstream_app.h"

#include <nvbufsurface.h>


#include <gst/rtsp-server/rtsp-server.h>

GST_DEBUG_CATEGORY (NVDS_APP);

#define MAX_DISPLAY_LEN 64

#define INOTIFY_EVENT_SIZE    (sizeof (struct inotify_event))
#define INOTIFY_EVENT_BUF_LEN (1024 * ( INOTIFY_EVENT_SIZE + 16))

// #define PGIE_CLASS_ID_VEHICLE 0
// #define PGIE_CLASS_ID_PERSON 2
#define PGIE_CLASS_ID_FACE 2
#define PGIE_CLASS_ID_SIDEFACE 6

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

static gboolean quit = FALSE;
int sock_image = 0; // 发送数据
int sock_warning = 0;
static GMutex sendImageLock; // 发送图片锁
static cv::Mat sendImage; // 发送图片
static GMutex sendStatusLock; // 发送状态锁
static int sendStatus = 0; // 发送状态
GThread *sendImageThread = NULL;
GThread *sendStatusThread = NULL;
GMutex status_lock;

/************************ YOLO Status DEFINE *************************/

#define max(a, b) (a)>(b)? (a):(b)
#define FPS 10
int sock_status = 0;

struct XYWH {
    double x, y, w, h;

    XYWH() {
        x = 0;
        y = 0;
        w = 0;
        h = 0;
    }
};

struct XYXY {
    int xmin, ymin, xmax, ymax;

    XYXY() {
        xmin = 0;
        ymin = 0;
        xmax = 0;
        ymax = 0;
    }
};

struct Box {
    XYXY xyxy;
    double confidence;
    int cls;

    Box(int x_min, int y_min, int x_max, int y_max, double conf, int cls_) {
        xyxy.xmin = x_min;
        xyxy.ymin = y_min;
        xyxy.xmax = x_max;
        xyxy.ymax = y_max;
        confidence = conf;
        cls = cls_;
    }
};

struct Box_xywh {
    XYWH xywh;
    double confidence;
    int cls;
};

class Status_Circular_Queue {
private:
    int status_bucket[6] = {0};
    int capacity;
    int rear;
    int *data;
public:
    Status_Circular_Queue(int capacity) {
        this->capacity = capacity;
        this->rear = -1;
        this->data = (int * )malloc(sizeof(int) * capacity);
        memset(this->data, 0, sizeof(int) * capacity);
    }
    bool isTrue(int x) {
        return (x >= 0) && (x <= 5);
    }
    int push(int x) {
        g_mutex_lock(&status_lock);
        rear = (rear + 1) % capacity;
        data[rear] = x;
        status_bucket[x]++;
        g_print("%d jianjian\n",getFrontData());
        // if(status_bucket[getFrontData()]>=0)
        status_bucket[getFrontData()]--;
        data[(rear + capacity  - 3 * FPS) % capacity]=0;
        g_mutex_unlock(&status_lock);
        return rear;
    }

    int getFrontData(){
        int x =  data[(rear + capacity  - 3 * FPS) % capacity];
        return x;
    }

    int *getData() {
        return data;
    }

    int *getStatusBucket() {
        return status_bucket;
    }

    int getCapacity() {
        return capacity;
    }
};


XYWH xyxy2XYWH(XYXY xyxy, int wide, int height) {
    XYWH xywh;
    xywh.x = (xyxy.xmin + xyxy.xmax) / 2.0 / wide;
    xywh.y = (xyxy.ymin + xyxy.ymax) / 2.0 / height;
    xywh.w = (xyxy.xmax - xyxy.xmin) * 1.0 / wide;
    xywh.h = (xyxy.ymax - xyxy.ymin) * 1.0 / height;
    return xywh;
}

class YOLOStatus {
public:
    YOLOStatus() {
        cls_["close_eye"] = 0;
        cls_["close_mouth"] = 1;
        cls_["face"] = 2;
        cls_["open_eye"] = 3;
        cls_["open_mouth"] = 4;
        cls_["phone"] = 5;
        cls_["sideface"] = 6;

        status_prior["normal"] = 0;
        status_prior["closeeye"] = 1;
        status_prior["yawn"] = 3;
        status_prior["calling"] = 4;
        status_prior["turning"] = 2;

        condition = {0, 1, 4, 2, 3};
    }
    void DebugNum(){
        int *status_bucket = this->status_seq->getStatusBucket();
        int num = 0;
        for(int i=1;i<=5;i++){
            g_print("status: %d , %d\n",i,status_bucket[i]);
            num += status_bucket[i];
        }
        g_print("sum:%d\n",num);
    }
    void check(int  & status) {
        int *status_bucket = this->status_seq->getStatusBucket();

        for (int i = 1; i <= 4; i++) {
                if (status_bucket[i] + status_bucket[5] == 3 * FPS) {
                    std::cout <<std::endl<< "status:" << i << ",Drowsy Driving" << std::endl<<std::endl;
                    sendStatus += 10;
                }
            }
            g_print("\n"); 
            DebugNum();
    }

    int determine(cv::Mat img, std::vector<Box> boxes) {
        int wide = img.cols, height = img.rows;
        int status = 0;// 最终状态，默认为正常0
        XYWH driver;//司机正脸的xywh
        XYXY driver_xyxy; // 司机正脸的xyxy
        double driver_conf = 0;// 司机正脸的置信度
        XYWH sideface;//司机侧脸的xywh
        XYXY sideface_xyxy;//司机侧脸的xyxy
        double sideface_conf = 0;// 司机侧脸的置信度

        XYWH face; // 司机的脸，不管正侧
        XYXY face_xyxy; // 司机的脸，不管正侧

        XYWH phone; // 手机的xywh坐标        XYWH face; // 司机的脸，不管正侧
        XYWH openeye; // 睁眼的xywh坐标
        XYWH closeeye; // 闭眼的xywh坐标
        double openeye_score = 0; // 睁眼的置信度
        double closeeye_score = 0; // 闭眼的置信度
        std::vector<Box_xywh> eyes; //第一遍扫描眼睛列表
        XYWH mouth; // 嘴巴的xywh坐标
        int mouth_status = 0; // 嘴巴的状态，0为正常，1为张嘴
        std::vector<Box_xywh> mouths; //第一遍扫描嘴巴列表
        bool phone_flag = false; // 手机状态，false为正常，true为打电话
        bool face_flag = false;
        //处理boxes
        for (auto box: boxes) {//遍历boxes,
            XYXY xyxy = box.xyxy;
            XYWH xywh = xyxy2XYWH(xyxy, wide, height);
            double conf = box.confidence;
            int cls = box.cls;
            if (cls == cls_["face"]) {
                if (xywh.x > 0.5 && xywh.y > driver.y) {
                    // box中心在右侧 并且在driver下方
                    driver = xywh;
                    driver_xyxy = xyxy;
                    driver_conf = conf;
                    face_flag = true;
                }
            } else if (cls == cls_["sideface"]) {
                if (xywh.x > 0.5 && xywh.y > sideface.y) {
                    // box中心在右侧 并且在sideface下方
                    sideface = xywh;
                    sideface_xyxy = xyxy;
                    sideface_conf = conf;
                    face_flag = true;
                }
            } else if (cls == cls_["phone"]) {
                if (xywh.x > 0.4 && xywh.y > 0.2 && xywh.y > phone.y && xywh.x > phone.x) {
                    // box中心在右侧 并且在phone下方
                    phone = xywh;
                    phone_flag = true;
                }
            } else if (cls == cls_["open_eye"] || cls == cls_["close_eye"]) {
                Box_xywh eye;
                eye.xywh = xywh;
                eye.confidence = conf;
                eye.cls = cls;
                eyes.push_back(eye);
            } else if (cls == cls_["open_mouth"] || cls == cls_["close_mouth"]) {
                Box_xywh mouth;
                mouth.xywh = xywh;
                mouth.confidence = conf;
                mouth.cls = cls;
                mouths.push_back(mouth);
            }
        }
        if (!face_flag) {
            return 4;
        }
        face = driver;
        face_xyxy = driver_xyxy;
        if (fabs(driver.x - sideface.x) < 0.1 && fabs(driver.y - sideface.y) < 0.1) {
            if (driver_conf > sideface_conf) {
                status = max(status_prior["normal"], status);
                face = driver;
                face_xyxy = driver_xyxy;
            } else {
                status = max(status_prior["turning"], status);
                face = sideface;
                face_xyxy = sideface_xyxy;
            }
        } else if (sideface.x > driver.x) {
            status = max(status_prior["turning"], status);
            face = sideface;
            face_xyxy = sideface_xyxy;
        }
        if (face.w == 0) {
            status = max(status_prior["turning"], status);
        }
        if (fabs(face.x - phone.x) < 0.3 && fabs(face.y - phone.y) < 0.3 && phone_flag) {
            status = max(status_prior["calling"], status);
        }
        for (auto eye_i: eyes) {
            if (eye_i.xywh.x < face_xyxy.xmin * 1.0 / wide || eye_i.xywh.x > face_xyxy.xmax * 1.0 / wide ||
                eye_i.xywh.y < face_xyxy.ymin * 1.0 / height || eye_i.xywh.y > face_xyxy.ymax * 1.0 / height) {
                continue;
            }
            if (eye_i.cls == cls_["open_eye"]) {
                if (eye_i.xywh.x > openeye.x) {
                    openeye = eye_i.xywh;
                    openeye_score = eye_i.confidence;
                }
            } else if (eye_i.cls == cls_["close_eye"]) {
                if (eye_i.xywh.x > closeeye.x) {
                    closeeye = eye_i.xywh;
                    closeeye_score = eye_i.confidence;
                }
            }
        }
        for (auto mouth_i: mouths) {
            if (mouth_i.xywh.x < face_xyxy.xmin * 1.0 / wide || mouth_i.xywh.x > face_xyxy.xmax * 1.0 / wide
                || mouth_i.xywh.y < face_xyxy.ymin * 1.0 / height || mouth_i.xywh.y > face_xyxy.ymax * 1.0 / height) {
                continue;
            }
            if (mouth_i.cls == cls_["open_mouth"]) {
                if (mouth_i.xywh.x > mouth.x) {
                    mouth = mouth_i.xywh;
                    mouth_status = 1;
                }
            } else if (mouth_i.cls == cls_["close_mouth"]) {
                if (mouth_i.xywh.x > mouth.x) {
                    mouth = mouth_i.xywh;
                    mouth_status = 0;
                }
            }
        }
        if (mouth_status == 1) {
            status = max(status_prior["yawn"], status);
        }
        if (fabs(closeeye.x - openeye.x) < 0.2) {
            if (closeeye_score > openeye_score) {
                status = max(status_prior["closeeye"], status);
            } else {
                status = max(status_prior["normal"], status);
            }
        } else {
            if (closeeye.x > openeye.x) {
                status = max(status_prior["closeeye"], status);
            } else {
                status = max(status_prior["normal"], status);
            }
        }
        return condition.at(status);
    }
    int *push_status(int status) {
        
        int loc = status_seq->push(status);
        int *data = status_seq->getData();
        check(status);
        g_print("status %d is pushed\n",status);
        return &data[loc];
    }
    void change_probe2definite(int status){
        g_mutex_lock(&status_lock);
        int * num = status_seq->getStatusBucket();  
        num[5]--;
        num[status]++;
        g_mutex_unlock(&status_lock);
    }

private:
    std::map<std::string, int> cls_;
    std::map<std::string, int> status_prior;
    std::vector<int> condition;
    Status_Circular_Queue *status_seq = new Status_Circular_Queue(10 * FPS);
};

YOLOStatus yoloStatus;
/********************** END OF YOLOSTATUS **************************/


// --------------------------- huawei cloud infer ----------------------------

struct ResponseData {
    std::string token;
};

// 回调函数来处理HTTP响应头部信息
size_t HeaderCallback(void* contents, size_t size, size_t nmemb, ResponseData* responseData) {
    size_t total_size = size * nmemb;
    std::string header(static_cast<char*>(contents), total_size);
    std::string token_key = "X-Subject-Token: ";

    size_t pos = header.find(token_key);
    if (pos != std::string::npos) {
        std::istringstream iss(header.substr(pos + token_key.length() - 1));
        std::getline(iss, responseData->token);
    }

    return total_size;
}

std::string get_huawei_cloud_token() {
    // 华为云认证服务的请求URL
    std::string auth_url = "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens";

    // 提供身份验证信息
    std::string auth_payload = "{\"auth\": {\"identity\": {\"methods\": [\"password\"],\"password\": {\"user\": {\"name\": \"test\",\"password\": \"qwer1234\",\"domain\": {\"name\": \"nobody_zgb\"}}}},\"scope\": {\"project\": {\"name\": \"cn-north-4\"}}}}";

    // 初始化CURL库
    CURL* curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Accept: application/json");

        // 设置CURL选项
        curl_easy_setopt(curl, CURLOPT_URL, auth_url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, auth_payload.c_str());

        ResponseData responseData;

        // 设置回调函数以处理响应头部信息
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &responseData);

        // 发送认证请求
        res = curl_easy_perform(curl);

        // 检查响应状态码
        if (res == CURLE_OK) {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

            if (response_code == 201) {
                curl_easy_cleanup(curl);
                curl_global_cleanup();
                return responseData.token;
            } else {
                std::cout << "认证失败，HTTP 状态码: " << response_code << std::endl;
            }
        } else {
            std::cout << "CURL 请求失败: " << curl_easy_strerror(res) << std::endl;
        }

        curl_easy_cleanup(curl);
        curl_global_cleanup();
    }

    return "";
}

size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *output) {
    size_t total_size = size * nmemb;
    output->append((char*)contents, total_size);
    return total_size;
}

void cloud_infer(cv::Mat DataImg, int *loc, int probe_status) {
    int tmp = 0;
    // Config url, token, and file path.
    std::string url = "https://2f5e8d5dc271407cb5d1c6b80a9e6a2c.apig.cn-north-4.huaweicloudapis.com/v1/infers/e8ab22a3-a4e1-4dc8-8adb-7a78fb2463d3";
    std::string token = get_huawei_cloud_token();
    cv::imwrite("1.jpg", DataImg);
    std::string file_path = "1.jpg";

    // Initialize libcurl.
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        // Set request headers.
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, ("X-Auth-Token: " + token).c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Set URL.
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Set POST data.
        curl_mime *mime;
        curl_mimepart *part;
        mime = curl_mime_init(curl);
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "images");
        curl_mime_filedata(part, file_path.c_str());

        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        std::string response_data;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);


        std::cout << "==========";
        // Perform the request.
        CURLcode res = curl_easy_perform(curl);

        // Check for errors.
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            std::cout << "Response Code: " << response_code << std::endl;
            std::cout << "Response Data: " << response_data << std::endl;

            // Parse JSON result
            Json::Value root;
            Json::Reader reader;
            bool parsingSuccessful = reader.parse(response_data, root);

            if (parsingSuccessful) {
                try{
                    float row = root["result"]["Roll"].asFloat();
                    float yaw = root["result"]["Yaw"].asFloat();
                    float pitch = root["result"]["Pitch"].asFloat();

                    if (row > -40 && row < 40 && yaw > -10) { // 没有转头
                        tmp = probe_status;
                        printf("row:%f, yaw:%f, pitch:%f\n ", row, yaw, pitch);
                        printf("n ");
                    } else { // 转头
                        tmp = 4;
                        printf("row:%f, yaw:%f, pitch:%f ", row, yaw, pitch);
                        printf("y");
                    }
                }
                catch (const std::exception& e) {
                    std::cout << "Error: " << e.what() << std::endl;
                }

            } else {
                std::cerr << "JSON parsing error: " << reader.getFormattedErrorMessages() << std::endl;
            }
        }

        // Clean up.
        curl_mime_free(mime);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    // Clean up libcurl.
    curl_global_cleanup();
    if(!*loc)
    {
        *loc = tmp;
        yoloStatus.change_probe2definite(*loc);
    }
    return;
}




gint frame_number = 0;
gchar pgie_classes_str[7][32] = {"close_eye", "close_mouth", "face", "open_eye", "open_mouth", "phone", "side_face"};
static gchar override_cfg_file[] = "config_infer_primary_yoloV5.txt";
static struct timeval ota_request_time; //OTA请求时间
static struct timeval ota_completion_time; //OTA完成时间


/* Config file parameters used for recording
 * User needs to change these parameters to reflect the change in recordings
 * e.g duration, start-time etc. */

/* Container format of recorded file 0 for mp4 and 1 for mkv format
 */
#define SMART_REC_CONTAINER 0

/* Cache functionality of recording
 */
#define CACHE_SIZE_SEC 15

/* Timeout if duration of recording is not set by user
 */
#define SMART_REC_DEFAULT_DURATION 10

/* Time at which it recording is started
 */
#define START_TIME 2

/* Duration of recording
 */
#define SMART_REC_DURATION 7

/* Interval in seconds for
 * SR start / stop events generation.
 */
#define SMART_REC_INTERVAL 7

static gboolean bbox_enabled = TRUE;
static gint enc_type = 0; // Default: Hardware encoder
static gint sink_type = 3; // Default: Eglsink
static guint sr_mode = 1; // Default: Video Only

//export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
//static gchar **override_cfg_file = NULL;

GOptionEntry entries[] = {
        {"bbox-enable",       'e', 0, G_OPTION_ARG_INT,            &bbox_enabled,
                "0: Disable bboxes, \
       1: Enable bboxes, \
       Default: bboxes enabled",                   NULL},
        {"enc-type",          'c', 0, G_OPTION_ARG_INT,            &enc_type,
                "0: Hardware encoder, \
       1: Software encoder, \
       Default: Hardware encoder", NULL},
        {"sink-type",         's', 0, G_OPTION_ARG_INT,            &sink_type,
                "1: Fakesink, \
       2: Eglsink, \
       3: RTSP sink, \
       Default: Eglsink",          NULL},
        {"override-cfg-file", 'o', 0, G_OPTION_ARG_FILENAME_ARRAY, &override_cfg_file,
                "Set the override config file, used for on-the-fly model update feature, \
                Default: config_infer_primary_yoloV5.txt",
                                                   NULL}// 这里不用动，之后如果测试成功再改
        ,
        {"sr-mode",           'm', 0, G_OPTION_ARG_INT,            &sr_mode,
                "SR mode: 0 = Audio + Video, \
       1 = Video only, \
       2 = Audio only",            NULL},
        {NULL},
};

typedef struct _OTAInfo {
    gchar *override_cfg_file;
//    AppCtx *appCtx; // 除去这个信息，用其他的代替
    GstElement *pgie;
} OTAInfo;

static GstElement *pipeline = NULL, *tee_pre_decode = NULL;
static NvDsSRContext *nvdssrCtx = NULL;
static GMainLoop *loop = NULL;


//send--flag
gboolean image_sendable = FALSE;
gboolean status_sendable = FALSE;

// websocket发送图片
void send_image(cv::Mat img) {
    
    std::vector<uchar> img_buffer;
    cv::imencode(".jpg", img, img_buffer);
    int img_size = img_buffer.size();
    int net_size = htonl(img_size);
    send(sock_image, &net_size, sizeof(int), 0);
    send(sock_image, img_buffer.data(), img_size, 0);
}

gpointer send_image_thread(gpointer data) {
    // g_print("send_image_thread start\n");
    while (1) {
        if (image_sendable == TRUE) {
            g_print("send_image in thread\n");
            g_mutex_lock(&sendImageLock);
            cv::Mat bk_image = sendImage.clone();
            g_mutex_unlock(&sendImageLock);
            send_image(bk_image);
            bk_image.release();
            image_sendable = FALSE;
            // g_print("send_image over\n");
        }
    }
}

void send_status(int status) {
    int net_num = htonl(status);  // 转换为网络字节序
    send(sock_status, &net_num, sizeof(int), 0);
}

gpointer send_status_thread(gpointer data) {
    // g_print("send_status_thread start\n");
    while (1) {
        if (status_sendable == TRUE) {
            // g_print("send status\n");

            g_mutex_lock(&sendStatusLock);
            int bk_status = sendStatus;
            g_mutex_unlock(&sendStatusLock);
            send_status(bk_status);
            status_sendable = FALSE;
            // g_print("send status over\n");
        }
    }
}



/*
 * 探针函数，用于在osd_sink_pad_buffer_probe中调用
 */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data) {
    GstBuffer *buf = (GstBuffer *) info->data;
    if (buf == NULL) {
        g_print("Unable to get GstBuffer\n");
    }
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    NvDsDisplayMeta *display_meta = NULL;
    NvOSD_RectParams *rect_params = NULL; //矩形框参数

    NvBufSurface *surface = NULL; // 用于获取图像数据
    GstMapInfo in_map_info; // 用于获取图像数据
    guint top = 0, left, width, height; // 人脸框的位置
    double face_conf = 0.0;//检测到的人脸/侧脸置信度
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        top =0;      
        //现在时间
        struct timeval now;
        int time = gettimeofday(&now, NULL);
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta * )(l_frame->data);
        std::vector<Box> boxes;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta * )(l_obj->data);
            Box box(obj_meta->rect_params.left, obj_meta->rect_params.top,
                    obj_meta->rect_params.left + obj_meta->rect_params.width,
                    obj_meta->rect_params.top + obj_meta->rect_params.height, obj_meta->confidence,
                    obj_meta->class_id);
            boxes.push_back(box);
            if (obj_meta->class_id == PGIE_CLASS_ID_FACE || obj_meta->class_id == PGIE_CLASS_ID_SIDEFACE) {
                // g_print("%d %d %d %d %d\n",box.xyxy.xmin,box.xyxy.xmax,box.xyxy.ymin,box.xyxy.ymax,box.cls);
                rect_params = &obj_meta->rect_params;
                top = rect_params->top;
                left = rect_params->left;
                width = rect_params->width;
                height = rect_params->height;
                face_conf = obj_meta->confidence;
                // g_print("face detected top: %d\n",top);
            }
        }


        guint batch_id = frame_meta->batch_id;
        if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {//获取图像数据
            g_print("gst_buffer_map() error!");
            gst_buffer_unmap(buf, &in_map_info);
            return GST_PAD_PROBE_OK;
        }
        surface = (NvBufSurface *) in_map_info.data;

        void *src_data = (char *) malloc(surface->surfaceList[frame_meta->batch_id].dataSize);

        if (src_data == NULL) {
            g_print("Error: failed to malloc src_data \n");
        }

        if (NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE) == 0) {
            NvBufSurfaceSyncForCpu(surface, -1, -1);
            uint32_t current_frame_width = surface->surfaceList[batch_id].width;
            uint32_t current_frame_height = surface->surfaceList[batch_id].height;
            if (surface->surfaceList[batch_id].mappedAddr.addr[0] == NULL) {
                g_print("NULL error!");
                continue;
            }
            cv::Mat current_frame_data = cv::Mat((gint) current_frame_height,
                                                 (gint) current_frame_width,
                                                 CV_8UC4,
                                                 surface->surfaceList[batch_id].mappedAddr.addr[0],
                                                 surface->surfaceList[batch_id].pitch
            );
            cv::Mat image_data((gint) current_frame_height,
                               (gint) current_frame_width,
                               CV_8UC4);
            current_frame_data.copyTo(image_data);
            g_mutex_lock(&sendImageLock);
            // RGBA convert BGR
            sendImage = cv::Mat((gint) current_frame_height,
                                (gint) current_frame_width,
                                CV_8UC3);
            cv::cvtColor(image_data, sendImage, cv::COLOR_RGBA2BGR);    // opencv4
            g_mutex_unlock(&sendImageLock);
            g_mutex_lock(&sendStatusLock);
            sendStatus = yoloStatus.determine(sendImage, boxes);
            g_mutex_unlock(&sendStatusLock);
            // g_print("image unlock\n");
            image_sendable = TRUE;//可以发送

            if (sendStatus!=3 &&top != 0 && face_conf < 0.4) {
                // 根据top left width height截取人脸
                cv::Rect rect(left, top, width, height);
                cv::Mat face = sendImage(rect);
                cv::imwrite("2.jpg", face);
                if(face.empty()){
                    g_print("FALSE!!!!!!!!\n\n\n\n");
                    exit(-1);
                }
                int *loc = yoloStatus.push_status(5), probe_status = sendStatus;
                // 调用云端来判别，这里孔雀空缺push_status
                g_print("call for cloud to infer\n");
                // 新开一个线程,用来运行云端的推理 cloud_landmarks_infer,并将线程独立出来
                std::thread cloud_landmark_infer_thread(cloud_infer, face, loc, probe_status);
                cloud_landmark_infer_thread.detach();
            } else {
                yoloStatus.push_status(sendStatus);
            }
            status_sendable = TRUE;
            NvBufSurfaceUnMap(surface, -1, -1);
        }
        gst_buffer_unmap(buf, &in_map_info);//释放资源
        if (src_data != NULL)
            free(src_data);
    }

    return GST_PAD_PROBE_OK;
}


static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}


static gpointer
smart_record_callback(NvDsSRRecordingInfo *info, gpointer userData) {
    static GMutex mutex;
    FILE *logfile = NULL;
    g_return_val_if_fail(info, NULL);

    g_mutex_lock(&mutex);
    logfile = fopen("smart_record.log", "a");
    if (logfile) {
        fprintf(logfile, "%d:%s:%d:%d:%s:%d channel(s):%d Hz:%ldms:%s:%s\n",
                info->sessionId, info->containsVideo ? "video" : "no-video",
                info->width, info->height, info->containsAudio ? "audio" : "no-audio",
                info->channels, info->samplingRate, info->duration,
                info->dirpath, info->filename);
        fclose(logfile);
    } else {
        g_print("Error in opeing smart record log file\n");
    }
    g_mutex_unlock(&mutex);

    return NULL;
}


static gboolean
smart_record_event_generator(gpointer data) {
    NvDsSRSessionId sessId = 0;
    NvDsSRContext *ctx = (NvDsSRContext *) data;
    guint startTime = START_TIME;
    guint duration = SMART_REC_DURATION;

    if (ctx->recordOn) {
        g_print("Recording done.\n");
        if (NvDsSRStop(ctx, 0) != NVDSSR_STATUS_OK)
            g_printerr("Unable to stop recording\n");
    } else {
        g_print("Recording started..\n");
        if (NvDsSRStart(ctx, &sessId, startTime, duration,
                        NULL) != NVDSSR_STATUS_OK)
            g_printerr("Unable to start recording\n");
    }
    return TRUE;
}

/**
 * Callback function to notify the status of the model update
 */
static void
infer_model_updated_cb(GstElement *gie, gint err, const gchar *config_file) { // 打印更新的信息~
    double otaTime = 0;
    gettimeofday(&ota_completion_time, NULL);

    otaTime = (ota_completion_time.tv_sec - ota_request_time.tv_sec) * 1000.0;
    otaTime += (ota_completion_time.tv_usec - ota_request_time.tv_usec) / 1000.0;

    const char *err_str = (err == 0 ? "ok" : "failed");
    g_print
            ("\nModel Update Status: Updated model : %s, OTATime = %f ms, result: %s \n\n",
             config_file, otaTime, err_str);
}


/**
 * Independent thread to perform model-update OTA process based on the inotify events
 * It handles currently two scenarios
 * 1) Local Model Update Request (e.g. Standalone Appliation)
 *    In this case, notifier handler watches for the ota_override_file changes
 * 2) Cloud Model Update Request (e.g. EGX with Kubernetes)
 *    In this case, notifier handler watches for the ota_override_file changes along with
 *    ..data directory which gets mounted by EGX deployment in Kubernetes environment.
 */

gpointer
ota_handler_thread(gpointer data) {// 顺便直接把pgie的指针传进来


    /*
     * 这里的逻辑有点像CPU的调度，在CPU调度中，如果有IO的操作
     * 那么采用的思路是让CPU去做其他的事情，等IO操作完成之后
     * 发送的信息被CPU接收到，CPU再去读取IO的文件。
     *
     * 这里其实是一样的，不用while(1)去循环查看当前的文件夹下面有没有文件的修改或者变动
     * 而是用Linux自带的inotify机制（说白了就是个函数），当文件夹下面有变动的时候，就会发送一个信号
     * 然后我们只需要去处理这个信号
     *
     * 在更换pgie的时候，只需要去把模型的文件路径传进去就好了
     * 也没有啥其他操作，希望不要寄了
     *
     * 参考的是deepstream_test5_app_main.c的文件，所以
     * 编译的时候
     * 得参考参考它的MAKEFILE
     * 如果编译没有通过可能需要小小的更改一下，应该问题不大
     * 希望如此
     * */
    int length, i = 0;
    char buffer[INOTIFY_EVENT_BUF_LEN]; // 用来存储Linux系统，当前的文件夹下面有什么变化 inotify机制
    OTAInfo *ota = (OTAInfo *) data;

    g_print("Now running ota_handler_thread\n");
    gchar *ota_ds_config_file = ota->override_cfg_file; //得到cfg的路径
    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n%s \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", ota_ds_config_file);
//    AppCtx *ota_appCtx = ota->appCtx;
    struct stat file_stat = {0};
    GstElement *primary_gie = NULL;
    gboolean connect_pgie_signal = FALSE;

    NvDsConfig override_config;
    //代替appCtx
    guint ota_inotify_fd = 0;
    guint ota_watch_desc = 0;


    ota_inotify_fd = inotify_init(); //初始化inotify机制，内部还是用的面向对象搞的（大雾）

    if (ota_inotify_fd < 0) {
        perror("inotify_init");
        return NULL;
    }

    char *real_path_ds_config_file = realpath(ota_ds_config_file, NULL); //得到真实的路径
    g_print("REAL PATH = %s\n", real_path_ds_config_file);

    gchar *ota_dir = g_path_get_dirname(real_path_ds_config_file); //得到config的路径
    ota_watch_desc =
            inotify_add_watch(ota_inotify_fd, ota_dir, IN_ALL_EVENTS); // 监视当前文件夹下面的所有事件

    int ret = lstat(ota_ds_config_file, &file_stat);
    ret = ret; // 也不知道干啥的

    if (S_ISLNK(file_stat.st_mode)) { //传入的应该都是绝对路径，所以这里应该不会出现软链接的情况，但是予以保留，防止Bug
        printf(" Override File Provided is Soft Link\n");
        gchar *parent_ota_dir = g_strdup_printf("%s/..", ota_dir);
        ota_watch_desc =
                inotify_add_watch(ota_inotify_fd, parent_ota_dir,
                                  IN_ALL_EVENTS);
    }

    while (1) {
        i = 0;
        length = read(ota_inotify_fd, buffer, INOTIFY_EVENT_BUF_LEN); //读取当前文件夹下面的变化的长度（？应该是次数

        if (length < 0) {
            perror("read");
        }

        if (quit == TRUE)
            goto done;

        while (i < length) {
            struct inotify_event *event = (struct inotify_event *) &buffer[i]; // 把时间提取出来

            if (connect_pgie_signal == FALSE) { //表示还没有连接到pgie
                primary_gie = ota->pgie; // 得到pgie的指针
                if (primary_gie) {
                    g_signal_connect(G_OBJECT(primary_gie), "model-updated",
                                     G_CALLBACK(infer_model_updated_cb), NULL); // 增加一个监听器(python里面对应的pad)
                    // 如果模型更新了，就会调用infer_model_updated_cb这个函数。用来打印更新信息
                    connect_pgie_signal = TRUE; // 已经连接上pgie(找到指针)，防止重复连接
                } else {
                    g_print
                            ("Gstreamer pipeline element nvinfer is yet to be created or invalid\n");
                    continue;
                }
            }

            if (event->len) {
                if (event->mask & IN_MOVED_TO) { // 处理文件移动到监视目录的事件
                    if (strstr("..data", event->name)) { // 如果文件名中包含data? 不是特别清湖这个来干嘛的
                        // 不知道这个是什么意思，还需要后续的理解 有可能是权重更改没更改文件的时候会发生这个事情
                        memset(&override_config, 0,
                               sizeof(override_config)); // 传入cfg文件
                        gettimeofday(&ota_request_time, NULL); // 得到当前时间
                        g_print("\nNew Model Update Request %s ----> %s\n",
                                GST_ELEMENT_NAME(primary_gie), override_cfg_file);

                        g_object_set(G_OBJECT(primary_gie),
                                     "config-file-path", override_cfg_file, NULL);
                    }
                }
                if (event->mask & IN_CLOSE_WRITE) { // 表示文件在写入并关闭后
                    if (!(event->mask & IN_ISDIR)) { //判断事件不是目录的事件（IN_ISDIR标志）。这是为了确保不是目录被写入，而是文件被写入
                        if (strstr(ota_ds_config_file, event->name)) { // 看是不是这个配置文件被更改了
                            g_print("File %s modified.\n", event->name);
                            memset(&override_config, 0,
                                   sizeof(override_config)); // same操作
                            gettimeofday(&ota_request_time, NULL);
                            g_print("\nNew Model Update Request %s ----> %s\n",
                                    GST_ELEMENT_NAME(primary_gie), override_cfg_file);
//                                g_object_set (G_OBJECT (primary_gie), "model-engine-file",
//                                              "model_b1_gpu0_fp32.engine", NULL); // 直接设置新的模型的路径??????????????
                            g_object_set(G_OBJECT(primary_gie),
                                         "config-file-path", override_cfg_file, NULL);
                        }
                    }
                }
            }
            i += INOTIFY_EVENT_SIZE + event->len;
        }
    }


    done: // 释放资源，下面都是复制过来的
    inotify_rm_watch(ota_inotify_fd, ota_watch_desc);

    close(ota_inotify_fd);

    free(real_path_ds_config_file);
    g_free(ota_dir);

    g_free(ota);
    return NULL;
}


gpointer
sr_thread(gpointer data) {
    int PORT = 4333;
    int BUFFER_SIZE = 1024;
    int server_fd, client_fd;
    struct sockaddr_in server_address, client_address;
    socklen_t client_len;
    char buffer[BUFFER_SIZE];

    // 创建 socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Failed to create socket");
        exit(EXIT_FAILURE);
    }

    // 设置 sockaddr_in 结构体
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(PORT);

    // 绑定 socket
    if (bind(server_fd, (struct sockaddr *) &server_address, sizeof(server_address)) < 0) {
        perror("Bind failed!!!!");
        exit(EXIT_FAILURE);
    }

    // 开始监听
    if (listen(server_fd, 10) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("Listening on port %d...\n", PORT);

    client_len = sizeof(client_address);

    // 等待连接
    if ((client_fd = accept(server_fd, (struct sockaddr *) &client_address, &client_len)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    printf("Connection established.\n");

    while (1) {
        memset(buffer, 0, BUFFER_SIZE);

        // 读取客户端消息
        ssize_t bytes_read = read(client_fd, buffer, BUFFER_SIZE);
        if (bytes_read <= 0) {
            printf("Connection closed or error\n");
            close(client_fd);
            exit(EXIT_SUCCESS);
        }

        // 检查是否收到了 'record'
        if (strncmp(buffer, "record", 6) == 0) {
            printf("Now start recording....\n\n\n\n\n\n\n\n");
            smart_record_event_generator(data);
            printf("Record generated! \n\n\n\n\n\n\n\n");
        }
    }

    close(client_fd);
    return 0;
}


int main(int argc, char *argv[]) {
    g_mutex_init(&sendImageLock);
    g_mutex_init(&sendStatusLock);
    g_mutex_init(&status_lock);
    //----------------------- connect the image_websocket---------------------------------
    struct sockaddr_in serv_addr;
    // 创建套接字
    if ((sock_image = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Socket creation error\n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8765);

    // 设置IP地址
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        printf("Invalid address / Address not supported\n");
        return -1;
    }

    // 连接
    if (connect(sock_image, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connection_image failed\n");
        return -1;
    }
    //---------------------- connection_image established -------------------------------------
    //----------------------- connect the status_websocket---------------------------------
    struct sockaddr_in serv_addr_status;
    // 创建套接字
    if ((sock_status = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("Socket creation error\n");
        return -1;
    }
    serv_addr_status.sin_family = AF_INET;
    serv_addr_status.sin_port = htons(8764);
    //设置IP地址
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr_status.sin_addr) <= 0) {
        printf("Invalid address / Address not supported\n");
        return -1;
    }
    // 连接
    if (connect(sock_status, (struct sockaddr *) &serv_addr_status, sizeof(serv_addr_status)) < 0) {
        printf("Connection_status failed\n");
        return -1;
    }
    //---------------------- connection_status established -------------------------------------
    // //----------------------- connect the warning_websocket---------------------------------
    // struct sockaddr_in serv_addr_warning;
    // // 创建套接字
    // if ((sock_warning = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    //     printf("Socket creation error\n");
    //     return -1;
    // }
    // serv_addr_warning.sin_family = AF_INET;
    // serv_addr_warning.sin_port = htons(8763);
    // //设置IP地址
    // if (inet_pton(AF_INET, "127.0.0.1", &serv_addr_warning.sin_addr) <= 0) {
    //     printf("Invalid address / Address not supported\n");
    //     return -1;
    // }
    // // 连接
    // if (connect(sock_warning, (struct sockaddr *) &serv_addr_warning, sizeof(serv_addr_warning)) < 0) {
    //     printf("Connection_status failed\n");
    //     return -1;
    // }
    // //---------------------- connection_status established -------------------------------------
    GstElement *source = NULL, *nvvidconv_src = NULL, *tee_pre = NULL,
            *queue_tee = NULL, *caps_nvvidconv_src = NULL, *streammux = NULL,
            *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL, *nvvidconv2 = NULL,
            *cap_filter = NULL, *sink = NULL, *queue_sr = NULL, *caps_sr = NULL,
            *encoder_pre = NULL, *parser_pre = NULL, *tes_fake = NULL, *tes_fake2 = NULL;

    GstCaps *caps = NULL, *caps_src = NULL, *caps_sr_dat = NULL;
    GstPad *osd_sink_pad = NULL;

    OTAInfo *otaInfo = NULL; // OTA info structure
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    guint i = 0, num_sources = 1;

    guint pgie_batch_size = 0;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    GstElement *transform = NULL;
    GOptionContext *gctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;

    NvDsSRInitParams params = {0};

    gctx = g_option_context_new("Nvidia DeepStream Test-SR app");
    group = g_option_group_new("SR_test", NULL, NULL, NULL, NULL);
    g_option_group_add_entries(group, entries);

    g_option_context_set_main_group(gctx, group);
    g_option_context_add_group(gctx, gst_init_get_option_group());

    GST_DEBUG_CATEGORY_INIT(NVDS_APP, "NVDS_APP", 0, NULL);

    if (!g_option_context_parse(gctx, &argc, &argv, &error)) {
        g_printerr("%s", error->message);
        g_print("%s", g_option_context_get_help(gctx, TRUE, NULL));
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new("dstest-sr-pipeline");

    source = gst_element_factory_make("nvarguscamerasrc", "src-elem");

    g_object_set(G_OBJECT(source), "bufapi-version", TRUE, NULL);

    nvvidconv_src = gst_element_factory_make("nvvideoconvert", "nvvidconv_src");

    /* Create tee which connects decoded source data and Smart record bin without bbox */
    tee_pre = gst_element_factory_make("tee", "tee-pre");

    queue_tee = gst_element_factory_make("queue", "queue-tee");

    GstElement *rate = gst_element_factory_make("videorate", "rate");
    g_object_set(G_OBJECT(rate), "drop-only", TRUE, NULL);
    g_object_set(G_OBJECT(rate), "max-rate",10, NULL);
    g_object_set(G_OBJECT(rate), "min-rate", 10, NULL);


    /* for pgie, before streammux */
    caps_src = gst_caps_from_string("video/x-raw(memory:NVMM), width=1280, height=720");
    caps_nvvidconv_src =
            gst_element_factory_make("capsfilter", "caps-nvvidconv-src");
    g_object_set(G_OBJECT(caps_nvvidconv_src), "caps", caps_src, NULL);
    gst_caps_unref(caps_src);

    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

    /* Use nvinfer to infer on batched frame. */
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

    /* Use convertor to convert from RGBA to CAPS filter data format */
    nvvidconv2 =
            gst_element_factory_make("nvvideoconvert", "nvvideo-converter2");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    /* Create a fakesink for test */

    tes_fake = gst_element_factory_make("fakesink", "fakesink-test");

    /* Finally render the osd output */
    if (prop.integrated) {
        transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
        if (!transform) {
            g_printerr("One tegra element could not be created. Exiting.\n");
            return -1;
        }
    }

    g_object_set(G_OBJECT(streammux), "live-source", 1, NULL);

    g_object_set(G_OBJECT(streammux), "batch-size", num_sources, NULL);

    g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                 MUXER_OUTPUT_HEIGHT,
                 "batched-push-timeout", 0, NULL);

    caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
    cap_filter =
            gst_element_factory_make("capsfilter", "src_cap_filter_nvvidconv");
    g_object_set(G_OBJECT(cap_filter), "caps", caps, NULL);
    gst_caps_unref(caps);

    GstElement *rtppay = NULL, *udpsink = NULL;

    GstElement *tee_after = gst_element_factory_make("tee", "tee-after");

    GstElement *queue_output = gst_element_factory_make("queue", "queue-output");
    GstElement *queue_record = gst_element_factory_make("queue", "queue-record");

    rtppay = gst_element_factory_make("rtph264pay", "rtppay");

    udpsink = gst_element_factory_make("udpsink", "udpsink");
    g_object_set(G_OBJECT(udpsink), "host", "224.224.255.255", "port", 5400, "async", FALSE, "sync", FALSE, NULL);


    int updsink_port_num = 5400;
    // Create an RTSP server
    GstRTSPServer *server = gst_rtsp_server_new();
    if (!server) {
        g_printerr("Failed to create RTSP server\n");
        return -1;
    }

    // Set server properties and attach it
    g_object_set(server, "service", "8554", NULL);

    // Create an RTSP media factory
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    if (!factory) {
        g_printerr("Failed to create RTSP media factory\n");
        return -1;
    }

    gchar launchStr[200] = {0};

    sprintf(launchStr,
            "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 \" )",
            updsink_port_num);

    gst_rtsp_media_factory_set_launch(factory, launchStr);

    // Add the factory to the server's mount points
    GstRTSPMountPoints *mountPoints = gst_rtsp_server_get_mount_points(server);
    gst_rtsp_mount_points_add_factory(mountPoints, "/test", factory);

    gst_rtsp_server_attach(server, NULL);
    g_print
            ("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n",
             8554);

    /* Configure the nvinfer element using the nvinfer config file. */
    g_object_set(G_OBJECT(pgie),
                 "config-file-path", "config_infer_primary_yoloV5.txt", NULL);
    g_print("\npgie name: %s \n",
            GST_ELEMENT_NAME(pgie));

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
        g_printerr
                ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
                 pgie_batch_size, num_sources);
        g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
    }

    g_object_set(G_OBJECT(nvosd), "process-mode", OSD_PROCESS_MODE,
                 "display-text", OSD_DISPLAY_TEXT, NULL);

    g_object_set(G_OBJECT(sink), "qos", 1, NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Set up the pipeline
     * source-> nvvidconv_src -> tee_pre -> queue_tee -> caps_nvvidconv_src -> streammux -> pgie -> nvvidconv -> nvosd -> nvvidconv2 -> caps_filter -> video-renderer
     *                                  |-> queue_sr -> caps_sr -> encoder -> parser -> recordbin
     */
    gst_bin_add_many(GST_BIN(pipeline), source, nvvidconv_src, tee_pre, tee_after, rate,
                     queue_tee, caps_nvvidconv_src, streammux, pgie, queue_output, queue_record,
                     nvvidconv, nvosd, nvvidconv2, cap_filter, rtppay, udpsink, tes_fake, NULL);

    if (prop.integrated) {
        gst_bin_add(GST_BIN(pipeline), transform);
    }

    /* Link the elements together till caps_nvvidconv_src */
    if (!gst_element_link_many(source, nvvidconv_src, tee_pre, rate,
                               queue_tee, caps_nvvidconv_src, NULL)) {
        g_printerr("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }


    /* Link decoder with streammux */
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad(caps_nvvidconv_src, pad_name_src);
    if (!srcpad) {
        g_printerr("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    /* Link the remaining elements of the pipeline to streammux */
    gst_element_link_many(streammux, pgie, nvvidconv, nvosd, tes_fake, NULL);

    /* Parameters are set before creating record bin
     * User can set additional parameters e.g recorded file path etc.
     * Refer NvDsSRInitParams structure for additional parameters
     */
    // params.containerType = SMART_REC_CONTAINER;
    params.containerType = NvDsSRContainerType::NVDSSR_CONTAINER_MP4;
    params.cacheSize = CACHE_SIZE_SEC;
    params.defaultDuration = SMART_REC_DEFAULT_DURATION;
    params.callback = smart_record_callback;
    params.fileNamePrefix = "SmartRecord";
    //    params.fileNamePrefix = bbox_enabled ? "With_BBox" : "Without_BBox";

    if (NvDsSRCreate(&nvdssrCtx, &params) != NVDSSR_STATUS_OK) {
        g_printerr("Failed to create smart record bin");
        return -1;
    }

    gst_bin_add_many(GST_BIN(pipeline), nvdssrCtx->recordbin, NULL);

    //--------------------------------------------------------------------------------

    /* for SR, after tee_pre */
    caps_sr_dat = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
    caps_sr =
            gst_element_factory_make("capsfilter", "caps-sr");
    g_object_set(G_OBJECT(caps_sr), "caps", caps_sr_dat, NULL);
    gst_caps_unref(caps_sr_dat);

    /* using hardware to accelerate */
    encoder_pre = gst_element_factory_make("nvv4l2h264enc", "encoder-pre");
    g_object_set(G_OBJECT(encoder_pre), "insert-sps-pps", 1, NULL);
    g_object_set(G_OBJECT(encoder_pre), "maxperf-enable", 1, NULL);
    g_object_set(G_OBJECT(encoder_pre), "bitrate", 4000000, NULL);

    /* Parse the encoded data after osd component */

    parser_pre = gst_element_factory_make("h264parse", "parser-pre");

    /* Use queue to connect the tee_post_osd to nvencoder */
    queue_sr = gst_element_factory_make("queue", "queue-sr");


    gst_bin_add_many(GST_BIN(pipeline), queue_sr, caps_sr, encoder_pre,
                     parser_pre, NULL);

    //    gst_element_link_many(nvvidconv_src, queue_sr, caps_sr, tes_fake2, NULL); // 先把这个tee给堵上，看来是其他地方出问题了

    if (!gst_element_link_many(tee_pre, queue_sr, caps_sr, encoder_pre, tee_after, queue_record,
                               parser_pre, nvdssrCtx->recordbin, NULL)) {
        g_print("Elements not linked. Exiting. \n");
        return -1;
    }

    if (!gst_element_link_many(tee_after, queue_output, rtppay, udpsink, NULL)) {
        g_print("Elements not linked. Exiting. \n");
        return -1;
    }
    //--------------------------------------------------------------------------------

    /*
     * Add probe to get the buffer flow
     */
    osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad)
        g_print("Unable to get sink pad\n");
    else
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);

    GThread *sr_threadd = NULL;
    if (nvdssrCtx) {
        //    if(0){
        // g_timeout_add (SMART_REC_INTERVAL * 1000, smart_record_event_generator,
        //                nvdssrCtx);
        sr_threadd = g_thread_new("sr_thread", sr_thread, nvdssrCtx);
    } else {
        printf("Smart video record thread unable to run!\n");
        exit(-1);
    }

    // make the ota module!

    otaInfo = (OTAInfo *) g_malloc0(sizeof(OTAInfo)); // 开辟OTA的数据结构空间
    //      otaInfo->appCtx = appCtx[i];
    otaInfo->override_cfg_file = override_cfg_file; // 把文件名传进去
    otaInfo->pgie = pgie;
    //    = "config_infer_primary_yoloV5.txt";
    GThread *ota_thread = g_thread_new("ota-handler-thread",
                                       ota_handler_thread, otaInfo);
    sendImageThread = g_thread_new(NULL, send_image_thread, NULL);
    sendStatusThread = g_thread_new(NULL, send_status_thread, NULL);
    /*
     * 使用g_thread_new函数创建线程后
     * 线程会立即开始运行指定的线程函数
     * 不需要显式地触发线程的运行
     * 操作系统会自动安排线程的调度
     * */
    /* Set the pipeline to "playing" state */
    g_print("Now using csi camera as input\n");

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print("Running...\n");
    g_main_loop_run(loop);
    if (pipeline && nvdssrCtx) {
        if (NvDsSRDestroy(nvdssrCtx) != NVDSSR_STATUS_OK)
            g_printerr("Unable to destroy recording instance\n");
    }
    /* Out of the main loop, clean up nicely */
    g_print("Returned, stopping playback\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Deleting pipeline\n");
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    g_mutex_clear(&sendImageLock);
    return 0;
}
