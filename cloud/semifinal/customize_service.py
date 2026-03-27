from __future__ import annotations

import gc
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, Optional, Union

import yolo.yolo_divide_and_conquer
from model_service.pytorch_model_service import PTServingBaseService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from yolo.yolo import *


class PTVisionService(PTServingBaseService):
    """ModelArts inference service for fatigue driving detection (semifinal).

    Receives an uploaded video via HTTP POST, runs the YOLO divide-and-conquer
    temporal localization pipeline, and returns detected dangerous driving
    behavior periods with categories and durations.

    This service follows the ModelArts lifecycle:
    ``_preprocess`` -> ``_inference`` -> ``_postprocess``.

    Attributes:
        capture: Path to the temporary video file, or None if not yet loaded.
        model_name: Name of the deployed model.
        model_path: Filesystem path to the model weights.
    """

    def __init__(self, model_name: str, model_path: str) -> None:
        """Initialize the service without loading model weights.

        The actual model is loaded lazily inside the YOLO inference module.

        Args:
            model_name: Name of the model registered in ModelArts.
            model_path: Path to the model weight file on disk.
        """
        self.capture: Optional[str] = None  # Fixed: set via tempfile in _preprocess to avoid race condition
        self.model_name = model_name
        self.model_path = model_path

    def _inference(self, data: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        """Run YOLO divide-and-conquer inference on the uploaded video.

        Triggers garbage collection before inference to free memory from
        prior requests, then delegates to the YOLO temporal localization
        pipeline.

        Args:
            data: Preprocessing result (the string ``'ok'`` on success,
                or an error dict on failure).

        Returns:
            A dict with structure::

                {
                    "result": {
                        "duration": <int>,   # inference time in ms
                        "drowsy": [          # detected behavior periods
                            {"periods": [start_ms, end_ms], "category": <int>},
                            ...
                        ]
                    }
                }
        """
        gc.collect()
        result = yolo.yolo_divide_and_conquer.yolo_run(source=self.capture)
        return result

    def _preprocess(self, data: Dict[str, Dict[str, Any]]) -> Union[str, Dict[str, str]]:
        """Save the uploaded video to a unique temporary file.

        Writes the video content from the HTTP request to a
        ``NamedTemporaryFile`` to avoid race conditions under concurrent
        requests.

        Args:
            data: Nested dict from the ModelArts framework with structure
                ``{key: {filename: file_object}}``.

        Returns:
            The string ``'ok'`` on success, or a dict with an error
            message on failure.
        """
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        tmp = tempfile.NamedTemporaryFile(
                            suffix=".mp4", delete=False
                        )  # Fixed: use tempfile to avoid race condition
                        self.capture = tmp.name
                        tmp.close()
                        with open(self.capture, "wb") as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}
                except Exception:
                    return {"message": "There was an error processing the file"}
        return "ok"

    def _postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up the temporary video file and return results.

        Args:
            data: The inference result dict from ``_inference``.

        Returns:
            The inference result dict, passed through unchanged.
        """
        if self.capture and os.path.exists(self.capture):  # Fixed: clean up temp file
            os.remove(self.capture)
        return data
