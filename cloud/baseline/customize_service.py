from __future__ import annotations

from PIL import Image
import copy
import sys
import traceback
import os
import numpy as np
import time
import cv2
from input_reader import InputReader
from tracker import Tracker
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from typing import Any, Dict, List, Optional, Union

from models.experimental import attempt_load
from utils1.general import check_img_size
from tempfile import NamedTemporaryFile
from utils1.torch_utils import TracedModel
from detect import detect
from model_service.pytorch_model_service import PTServingBaseService


class fatigue_driving_detection(PTServingBaseService):
    """Baseline ModelArts service for fatigue driving detection.

    Processes uploaded dashcam videos to detect dangerous driving behaviors
    using YOLOv5 for object detection (face, phone) and OpenSeeFace facial
    landmark tracking for eye/mouth state and head pose estimation.

    Detects four categories of fatigue/distraction:
        1. Eyes closed for >= 3 seconds
        2. Yawning (mouth open) for >= 3 seconds
        3. Phone use for >= 3 seconds
        4. Looking around (head turned) for >= 3 seconds

    This service follows the ModelArts lifecycle:
    ``_preprocess`` -> ``_inference`` -> ``_postprocess``.

    Attributes:
        model_name: Name of the deployed model.
        model_path: Filesystem path to the model weights.
        capture: Path to the temporary video file, or None.
        width: Expected video width in pixels.
        height: Expected video height in pixels.
        fps: Expected video frame rate.
        model: The traced YOLOv5 model for phone/face detection.
        tracker: OpenSeeFace tracker for facial landmark detection.
    """

    def __init__(self, model_name: str, model_path: str) -> None:
        """Initialize the detection service and load model weights.

        Loads the YOLOv5 model for face/phone detection and initializes
        the OpenSeeFace facial landmark tracker for 68-point landmarks.

        Args:
            model_name: Name of the model registered in ModelArts.
            model_path: Path to the YOLOv5 ``.pt`` weight file.
        """
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path

        self.capture: Optional[str] = None  # Fixed: set via tempfile in _preprocess to avoid race condition

        self.width: int = 1920
        self.height: int = 1080
        self.fps: int = 30
        # Fixed: removed unused self.first = True

        self.standard_pose: List[int] = [180, 40, 80]
        self.look_around_frame: int = 0
        self.eyes_closed_frame: int = 0
        self.mouth_open_frame: int = 0
        self.use_phone_frame: int = 0
        # lStart, lEnd) = (42, 48)
        self.lStart: int = 42
        self.lEnd: int = 48
        # (rStart, rEnd) = (36, 42)
        self.rStart: int = 36
        self.rEnd: int = 42
        # (mStart, mEnd) = (49, 66)
        self.mStart: int = 49
        self.mEnd: int = 66
        self.EYE_AR_THRESH: float = 0.2
        self.MOUTH_AR_THRESH: float = 0.6
        self.frame_3s: int = self.fps * 3
        self.face_detect: int = 0

        self.weights: str = "best.pt"
        self.imgsz: int = 640

        self.device: str = 'cpu'  # 大赛后台使用CPU判分

        model = attempt_load(model_path, map_location=self.device)
        self.stride: int = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        self.model = TracedModel(model, self.device, self.imgsz)


        self.need_reinit: int = 0
        self.failures: int = 0

        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=4, max_faces=4,
                          discard_after=10, scan_every=3, silent=True, model_type=3,
                          model_dir=None, no_gaze=False, detection_threshold=0.6,
                          use_retinaface=0, max_feature_updates=900,
                          static_model=True, try_hard=False)

        # self.temp = NamedTemporaryFile(delete=False)  # 用来存储视频的临时文件

    def _preprocess(self, data: Dict[str, Dict[str, Any]]) -> Union[str, Dict[str, str]]:
        """Save the uploaded video to a unique temporary file.

        Writes the video content from the HTTP request to a
        ``NamedTemporaryFile`` to avoid race conditions when handling
        concurrent requests.

        Args:
            data: Nested dict from the ModelArts framework with structure
                ``{key: {filename: file_object}}``.

        Returns:
            The string ``'ok'`` on success, or a dict with an error
            message on failure.
        """
        # preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        tmp = NamedTemporaryFile(suffix='.mp4', delete=False)  # Fixed: use tempfile to avoid race condition
                        self.capture = tmp.name
                        tmp.close()
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'

    def _inference(self, data: Union[str, Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """Run frame-by-frame fatigue detection on the uploaded video.

        Processes the video sequentially:

        1. Crops each frame to the driver's region (right 2/3 of frame).
        2. Runs YOLOv5 to detect face and phone bounding boxes.
        3. Uses OpenSeeFace tracker for 68-point facial landmarks.
        4. Computes Eye Aspect Ratio (EAR) for drowsiness detection.
        5. Computes Mouth Aspect Ratio (MAR) for yawning detection.
        6. Checks head pose Euler angles for distraction detection.
        7. Counts consecutive frames exceeding thresholds (3 seconds).

        Args:
            data: The result from ``_preprocess`` — either ``'ok'`` or
                an error dict.

        Returns:
            A dict with structure::

                {
                    "result": {
                        "category": <int>,   # 0-4 behavior category
                        "duration": <int>    # processing time in ms
                    }
                }

            Category codes: 0=normal, 1=eyes closed, 2=yawning,
            3=phone use, 4=looking around.
        """
        print(data)
        result: Dict[str, Dict[str, Any]] = {"result": {"category": 0, "duration": 6000}}

        self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps)
        source_name = self.input_reader.name
        now = time.time()
        while self.input_reader.is_open():
            if not self.input_reader.is_open() or self.need_reinit == 1:
                self.input_reader = InputReader(self.capture, 0, self.width, self.height, self.fps, use_dshowcapture=False, dcap=None)
                if self.input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {self.input_reader.name} instead of {source_name}.")
                    # sys.exit(1)
                self.need_reinit = 2
                time.sleep(0.02)
                continue
            if not self.input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = self.input_reader.read()

            self.need_reinit = 0

            try:
                if frame is not None:
                    # 剪裁主驾驶位
                    frame = frame[:, 600:1920, :]

                    # 检测驾驶员是否接打电话 以及低头的人脸
                    bbox = detect(self.model, frame, self.stride, self.imgsz)
                    # print(results)

                    for box in bbox:
                        if box[0] == 0:
                            self.face_detect = 1
                        if box[0] == 1:
                            self.use_phone_frame += 1

                    # 检测驾驶员是否张嘴、闭眼、转头
                    faces = self.tracker.predict(frame)
                    if len(faces) > 0:

                        face_num = 0
                        max_x = 0
                        for face_num_index, f in enumerate(faces):
                            if max_x <= f.bbox[3]:
                                face_num = face_num_index
                                max_x = f.bbox[3]

                        f = faces[face_num]
                        f = copy.copy(f)
                        # 检测是否转头
                        if np.abs(self.standard_pose[0] - f.euler[0]) >= 45 or np.abs(self.standard_pose[1] - f.euler[1]) >= 45 or \
                                np.abs(self.standard_pose[2] - f.euler[2]) >= 45:
                            self.look_around_frame += 1
                        else:
                            self.look_around_frame = 0

                        # 检测是否闭眼
                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = f.lms[self.lStart:self.lEnd]
                        rightEye = f.lms[self.rStart:self.rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        if ear < self.EYE_AR_THRESH:
                            self.eyes_closed_frame += 1
                        else:
                            self.eyes_closed_frame = 0
                        # print(ear, eyes_closed_frame)

                        # 检测是否张嘴
                        mar = mouth_aspect_ratio(f.lms)

                        if mar > self.MOUTH_AR_THRESH:
                            self.mouth_open_frame += 1
                        else:
                            self.mouth_open_frame = 0  # Fixed: reset counter when mouth is closed
#                         print(mar)

#                         print(len(f.lms), f.euler)
                    else:
                        if self.face_detect:
                            self.look_around_frame += 1
                            self.face_detect = 0
                    # print(self.look_around_frame)
                    if self.use_phone_frame >= self.frame_3s:
                        result['result']['category'] = 3
                        break

                    elif self.look_around_frame >= self.frame_3s:
                        result['result']['category'] = 4
                        break

                    elif self.mouth_open_frame >= self.frame_3s:
                        result['result']['category'] = 2
                        break

                    elif self.eyes_closed_frame >= self.frame_3s:
                        result['result']['category'] = 1
                        break
                    else:
                        result['result']['category'] = 0

                    self.failures = 0
                else:
                    break
            except KeyboardInterrupt:  # Fixed: KeyboardInterrupt is not a subclass of Exception; catch separately
                print("Quitting")
                break
            except Exception:
                traceback.print_exc()
                self.failures += 1
                if self.failures > 30:   # 失败超过30次就默认返回
                    break
            del frame
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        result['result']['duration'] = duration
        return result

    def _postprocess(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Clean up the temporary video file and return results.

        Args:
            data: The inference result dict from ``_inference``.

        Returns:
            The inference result dict, passed through unchanged.
        """
        if self.capture and os.path.exists(self.capture):  # Fixed: clean up temp file
            os.remove(self.capture)
        return data
