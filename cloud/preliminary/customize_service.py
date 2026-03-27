import gc
import logging
import os
import tempfile
import warnings

import yolo.yolo_divide_and_conquer
from model_service.pytorch_model_service import PTServingBaseService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from yolo.yolo import *


class PTVisionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.capture = None  # Fixed: set via tempfile in _preprocess to avoid race condition
        self.model_name = model_name
        self.model_path = model_path

    def _inference(self, data):
        gc.collect()
        result = yolo.yolo_divide_and_conquer.yolo_run(source=self.capture)
        return result

    def _preprocess(self, data):
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

    def _postprocess(self, data):
        if self.capture and os.path.exists(self.capture):  # Fixed: clean up temp file
            os.remove(self.capture)
        return data
