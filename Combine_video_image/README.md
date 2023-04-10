## Combination of video-classification and image-classification
### Quick Start
* Use nanodet + mobilenetV2 + cpu: 
    ```python
    python combine.py --video_model nanodet --image_model mobilenet --mouth_model ./mobilenet/MobileNetV2_yawnclass.pth --eye_model ./mobilenet/MobileNetV2_eyeclass.pth --device cpu
    ```

* Defualt run: nanodet + SVM + cpu:(目前有点bug)
    ```python
    python combine.py
    ```

* Parameter explanation:
    -  --video_model : choose nanodet(defualt) or yolo；For example：--video_model nanodet
    -  --image_model : choose SVM(defualt) or MobilenetV2;For example: --image_model svm
    -  --mouth_model : the path of mouth model of image-classification
    -  --eye_model : the path of eyes model of image-classification
    -  --devive : choose cpu(defualt) or gpu

### Need to do:
* Add Yolo : add code in class Combination , add model directory for yolo
* Testing the cost of multi-thread and serial
