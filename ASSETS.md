# Large Binary Assets

This repository uses [GitHub Releases](https://github.com/Nobody-Zhang/huaweicloud_2023/releases/tag/v1.0) to host large binary files that are not tracked in git.

## Quick Start

```bash
bash scripts/download_assets.sh
```

The script downloads all assets to their correct locations, skipping files that already exist.

## Asset Inventory

### Model Weights (186 MB)

| File | Size | Location |
|------|------|----------|
| `shape_predictor_68_face_landmarks.dat` | 95 MB | `Jetson TX2 NX Deployment/Code/Cloud_finetune/yolov5/` |
| `yolov5s.pt` | 14 MB | `Jetson TX2 NX Deployment/Code/OTA/` |
| `best (1).pt` | 4 MB | `Jetson TX2 NX Deployment/Code/OTA/` |
| `ONet.pt` | 1.5 MB | `Jetson TX2 NX Deployment/Code/OTA/mtcnn_landmarks/infer_models/` |
| `yolov5s_best.onnx` | 27 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `best.onnx` | 27 MB | `Jetson TX2 NX Deployment/Code/OTA/` |
| `mixed_n.onnx` | 7 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `yolov5n_best.onnx` | 7 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `best.bin` (OpenVINO) | 27 MB | `ModelArts .../Preliminary_BEST_bkup_cloud/.../fine_tune_openvino_model/` |
| `best.bin` (OpenVINO) | 27 MB | `ModelArts .../semi-final_BEST_bkup_cloud/.../fine_tune_openvino_model/` |

### TensorRT Engines (88 MB)

> These are platform-specific to Jetson TX2 NX. If your GPU architecture differs, regenerate from ONNX.

| File | Size | Location |
|------|------|----------|
| `v5s_mixes.engine` | 35 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `model_b1_gpu0_fp32_1.engine` | 13 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `model_b1_gpu0_fp32.engine` | 13 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `zgb_n.engine` | 13 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `model_b1_gpu0_v5n_mixed.engine` | 13 MB | `Jetson TX2 NX Deployment/Code/main/` |

### Python Wheels (77 MB unique, 4 copies each)

| File | Size | Locations |
|------|------|-----------|
| `scikit_learn-1.0.2-*.whl` | 24 MB | Preliminary, semi-final, OTA/mtcnn, mtcnn_landmarks |
| `scikit_image-0.19.3-*.whl` | 13 MB | Preliminary, semi-final, OTA/mtcnn, mtcnn_landmarks |

### Videos (26 MB)

| File | Size | Location(s) |
|------|------|-------------|
| `SmartRecord_00000_20231031-043454_8868.mp4` | 4 MB | `main/` |
| `SmartRecord_00000_20231031-072426_8890.mp4` | 4 MB | `main/` |
| `SmartRecord_00000_20231028-065356_8625.mp4` | 4 MB | `main/` |
| `SmartRecord_00001_20230830-080226_2871.mp4` | 4 MB | `OTA/videos/`, `APIGW/videos/` |
| `SmartRecord_00000_20230830-080212_2871.mp4` | 3 MB | `OTA/videos/`, `APIGW/videos/` |
| `day_man_053_31_1.mp4` | 1 MB | `OTA/videos/`, `APIGW/videos/` |
| `1.mp4` | 1 MB | `OTA/mtcnn_landmarks/test/` |

### Audio (36 MB)

| File | Size | Location |
|------|------|----------|
| `NWGYU.wav` | 36 MB | `main/vuertsp-master/src/assets/audio/` |

### Archives (103 MB)

| File | Size | Location |
|------|------|----------|
| `vuertsp-master.zip` | 68 MB | `Jetson TX2 NX Deployment/Code/main/` |
| `vue.zip` | 34 MB | `main/vuertsp-master/` |

### Documents (70 MB)

| File | Size | Location |
|------|------|----------|
| `生产队的大萝卜-答辩定稿PPT.pptx` | 66 MB | repo root |
| `生产队的大萝卜-技术文档.pdf` | 4 MB | repo root |

### Other (9 MB)

| File | Size | Location |
|------|------|----------|
| `Certificate.png` | 3 MB | repo root |
| `tmp.npy` | 6 MB | `main/` |
| `chunk-vendors.*.js.map` | 1 MB x2 | `dist/js/` |

## Total

- **40 tracked files** removed from git
- **~595 MB** total (with duplicates), **~380 MB** unique downloads
