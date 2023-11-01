//
// Created by cassius on 02/09/23.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdlib>
#include <unistd.h>

std::condition_variable c_v;
std::mutex mtx;
std::atomic<bool> stop_threads(false);

using namespace std;

bool check_camera() {
    const char* videoDevicePath = "/dev/video1";
    if (access(videoDevicePath, F_OK) != -1) {
        return true;
    } else {
        cout<<"the camera is gone\n";
        return false;
    }
}

bool check_network() {
    std::string host = "huaweicloud.com";
    std::string command = "ping -q -c 1 -W 1 " + host;
    int result = system(command.c_str());
    if (result == 0) return true;
    else return false;
}

void main_thread_function() {
    cv::VideoCapture cap(1);
    if(!cap.isOpened()){
	    std:: cout<<"unable to open the camera\n";
    }
    while (!stop_threads) {
        cv::Mat frame;
        cap >> frame;  // 读取摄像头的视频帧
        if (frame.empty()) {  // 检查是否成功读取视频帧
            printf("Unable to read frame from the webcam.\n");
            break;
        }

        cv::imshow("Webcam", frame);  // 显示视频帧窗口

        if (cv::waitKey(1) == 'q') {  // 按下 'q' 键退出循环
            break;
        }
        sleep(1);
    }

    cap.release();  // 释放摄像头
    cv::destroyAllWindows();  // 关闭视频帧窗口
}

int main() {
    std::thread camera_thread([&]() {
        while (!stop_threads) {
            if (!check_camera()) {
                std::unique_lock<std::mutex> lock(mtx);
                stop_threads = true;
                c_v.notify_all();
                break;
            }
        }
    });

    std::thread network_thread([&]() {
        while (!stop_threads) {
            if (!check_network()) {
                std::unique_lock<std::mutex> lock(mtx);
                stop_threads = true;
                c_v.notify_all();
                break;
            }
        }
    });

    std::thread main_thread(main_thread_function);

    {
        std::unique_lock<std::mutex> lock(mtx);
        c_v.wait(lock, [] { return stop_threads.load(); });
    }

    camera_thread.join();
    network_thread.join();
    main_thread.join();

    return 0;
}

