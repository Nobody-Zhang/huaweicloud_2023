#!/usr/bin/env bash
# download_assets.sh — Download large binary assets from GitHub Releases
# Usage: bash scripts/download_assets.sh
#
# Assets are hosted at:
#   https://github.com/Nobody-Zhang/huaweicloud_2023/releases/download/v1.0/
#
# Run from the repository root directory.

set -euo pipefail

BASE_URL="https://github.com/Nobody-Zhang/huaweicloud_2023/releases/download/v1.0"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

download() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ]; then
        echo "[SKIP] $dest (already exists)"
        return
    fi
    echo "[DOWN] $dest"
    curl -fSL --retry 3 -o "$dest" "$url"
}

echo "=== Model Weights ==="

download "$BASE_URL/shape_predictor_68_face_landmarks.dat" \
    "Jetson TX2 NX Deployment/Code/Cloud_finetune/yolov5/shape_predictor_68_face_landmarks.dat"

download "$BASE_URL/yolov5s.pt" \
    "Jetson TX2 NX Deployment/Code/OTA/yolov5s.pt"

download "$BASE_URL/best_1.pt" \
    "Jetson TX2 NX Deployment/Code/OTA/best (1).pt"

download "$BASE_URL/ONet.pt" \
    "Jetson TX2 NX Deployment/Code/OTA/mtcnn_landmarks/infer_models/ONet.pt"

download "$BASE_URL/yolov5s_best.onnx" \
    "Jetson TX2 NX Deployment/Code/main/yolov5s_best.onnx"

download "$BASE_URL/best.onnx" \
    "Jetson TX2 NX Deployment/Code/OTA/best.onnx"

download "$BASE_URL/mixed_n.onnx" \
    "Jetson TX2 NX Deployment/Code/main/mixed_n.onnx"

download "$BASE_URL/yolov5n_best.onnx" \
    "Jetson TX2 NX Deployment/Code/main/yolov5n_best.onnx"

download "$BASE_URL/best_openvino.bin" \
    "ModelArts automatic evaluation/Preliminary_BEST_bkup_cloud/Deployment_yolo/yolo/fine_tune_openvino_model/best.bin"

cp "ModelArts automatic evaluation/Preliminary_BEST_bkup_cloud/Deployment_yolo/yolo/fine_tune_openvino_model/best.bin" \
   "ModelArts automatic evaluation/semi-final_BEST_bkup_cloud/Deployment_yolo/yolo/fine_tune_openvino_model/best.bin" 2>/dev/null && \
   echo "[COPY] semi-final best.bin (identical to preliminary)" || true

echo ""
echo "=== TensorRT Engines (Jetson-specific, rebuild if architecture differs) ==="

download "$BASE_URL/v5s_mixes.engine" \
    "Jetson TX2 NX Deployment/Code/main/v5s_mixes.engine"

download "$BASE_URL/model_b1_gpu0_fp32_1.engine" \
    "Jetson TX2 NX Deployment/Code/main/model_b1_gpu0_fp32_1.engine"

download "$BASE_URL/model_b1_gpu0_fp32.engine" \
    "Jetson TX2 NX Deployment/Code/main/model_b1_gpu0_fp32.engine"

download "$BASE_URL/zgb_n.engine" \
    "Jetson TX2 NX Deployment/Code/main/zgb_n.engine"

download "$BASE_URL/model_b1_gpu0_v5n_mixed.engine" \
    "Jetson TX2 NX Deployment/Code/main/model_b1_gpu0_v5n_mixed.engine"

echo ""
echo "=== Python Wheels ==="

download "$BASE_URL/scikit_learn-1.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
    "ModelArts automatic evaluation/Preliminary_BEST_bkup_cloud/Deployment_yolo/scikit_learn-1.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

download "$BASE_URL/scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl" \
    "ModelArts automatic evaluation/Preliminary_BEST_bkup_cloud/Deployment_yolo/scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl"

# Copy wheels to other locations (identical files)
for dest_dir in \
    "ModelArts automatic evaluation/semi-final_BEST_bkup_cloud/Deployment_yolo" \
    "Jetson TX2 NX Deployment/Code/OTA/mtcnn_landmarks" \
    "Jetson TX2 NX Deployment/Code/mtcnn_landmarks"; do
    mkdir -p "$dest_dir"
    cp "ModelArts automatic evaluation/Preliminary_BEST_bkup_cloud/Deployment_yolo/scikit_learn-1.0.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
       "$dest_dir/" 2>/dev/null && echo "[COPY] $dest_dir/scikit_learn*.whl" || true
    cp "ModelArts automatic evaluation/Preliminary_BEST_bkup_cloud/Deployment_yolo/scikit_image-0.19.3-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl" \
       "$dest_dir/" 2>/dev/null && echo "[COPY] $dest_dir/scikit_image*.whl" || true
done

echo ""
echo "=== Videos ==="

download "$BASE_URL/SmartRecord_00000_20231031-043454_8868.mp4" \
    "Jetson TX2 NX Deployment/Code/main/SmartRecord_00000_20231031-043454_8868.mp4"

download "$BASE_URL/SmartRecord_00000_20231031-072426_8890.mp4" \
    "Jetson TX2 NX Deployment/Code/main/SmartRecord_00000_20231031-072426_8890.mp4"

download "$BASE_URL/SmartRecord_00000_20231028-065356_8625.mp4" \
    "Jetson TX2 NX Deployment/Code/main/SmartRecord_00000_20231028-065356_8625.mp4"

download "$BASE_URL/SmartRecord_00001_20230830-080226_2871.mp4" \
    "Jetson TX2 NX Deployment/Code/OTA/videos/SmartRecord_00001_20230830-080226_2871.mp4"

cp "Jetson TX2 NX Deployment/Code/OTA/videos/SmartRecord_00001_20230830-080226_2871.mp4" \
   "Jetson TX2 NX Deployment/Code/APIGW-python-sdk-2.0.4/videos/" 2>/dev/null && \
   echo "[COPY] APIGW SmartRecord_00001" || true

download "$BASE_URL/SmartRecord_00000_20230830-080212_2871.mp4" \
    "Jetson TX2 NX Deployment/Code/OTA/videos/SmartRecord_00000_20230830-080212_2871.mp4"

cp "Jetson TX2 NX Deployment/Code/OTA/videos/SmartRecord_00000_20230830-080212_2871.mp4" \
   "Jetson TX2 NX Deployment/Code/APIGW-python-sdk-2.0.4/videos/" 2>/dev/null && \
   echo "[COPY] APIGW SmartRecord_00000" || true

download "$BASE_URL/day_man_053_31_1.mp4" \
    "Jetson TX2 NX Deployment/Code/OTA/videos/day_man_053_31_1.mp4"

cp "Jetson TX2 NX Deployment/Code/OTA/videos/day_man_053_31_1.mp4" \
   "Jetson TX2 NX Deployment/Code/APIGW-python-sdk-2.0.4/videos/" 2>/dev/null && \
   echo "[COPY] APIGW day_man" || true

download "$BASE_URL/mtcnn_test_1.mp4" \
    "Jetson TX2 NX Deployment/Code/OTA/mtcnn_landmarks/test/1.mp4"

echo ""
echo "=== Audio ==="

download "$BASE_URL/NWGYU.wav" \
    "Jetson TX2 NX Deployment/Code/main/vuertsp-master/src/assets/audio/NWGYU.wav"

echo ""
echo "=== Archives ==="

download "$BASE_URL/vuertsp-master.zip" \
    "Jetson TX2 NX Deployment/Code/main/vuertsp-master.zip"

download "$BASE_URL/vue.zip" \
    "Jetson TX2 NX Deployment/Code/main/vuertsp-master/vue.zip"

echo ""
echo "=== Documents ==="

download "$BASE_URL/presentation.pptx" \
    "生产队的大萝卜-答辩定稿PPT.pptx"

download "$BASE_URL/technical_doc.pdf" \
    "生产队的大萝卜-技术文档.pdf"

echo ""
echo "=== Other ==="

download "$BASE_URL/Certificate.png" \
    "Certificate.png"

download "$BASE_URL/tmp.npy" \
    "Jetson TX2 NX Deployment/Code/main/tmp.npy"

download "$BASE_URL/chunk-vendors.466ac348.js.map" \
    "Jetson TX2 NX Deployment/Code/dist/js/chunk-vendors.466ac348.js.map"

download "$BASE_URL/chunk-vendors.13368598.js.map" \
    "Jetson TX2 NX Deployment/Code/dist/js/chunk-vendors.13368598.js.map"

echo ""
echo "=== Done ==="
echo "All assets downloaded. Total unique downloads: 28 files (+ 8 copies)"
