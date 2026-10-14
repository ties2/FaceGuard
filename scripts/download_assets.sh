#!/usr/bin/env bash
# Download face detector / recogniser weights (OpenCV Zoo) and the public face
# images used to build the bona fide corpus. Nothing here is committed to git.
set -euo pipefail
mkdir -p assets/models data/raw

ZOO=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
curl -sSL -o assets/models/face_detection_yunet_2023mar.onnx  "$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx"   # MIT
curl -sSL -o assets/models/face_recognition_sface_2021dec.onnx "$ZOO/face_recognition_sface/face_recognition_sface_2021dec.onnx" # Apache-2.0

# Public face photos (MIT-licensed repos; faces only, masks excluded)
if [ ! -d data/raw/face-mask-detection ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/chandrikadeb7/Face-Mask-Detection data/raw/face-mask-detection
  git -C data/raw/face-mask-detection sparse-checkout set dataset/without_mask
fi
if [ ! -d data/raw/deepface ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/serengil/deepface data/raw/deepface
  git -C data/raw/deepface sparse-checkout set tests/unit/dataset
fi
echo "assets ready"
