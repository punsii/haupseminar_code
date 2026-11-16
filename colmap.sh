#!/usr/bin/env bash
# Work out where each photo in a scene folder was taken from, using COLMAP.
#
#   usage: ./colmap.sh [scene_dir]
#          scene_dir  default ../data/scene2, must contain images/
#
#   USE_GPU=1 ./colmap.sh ...   much faster feature matching on an NVIDIA machine;
#                               the default 0 is for the AMD laptop
#
# Produces <scene>/undistorted/{images,sparse}, with sparse/ in text form, which is what
# src/realtime_training reads. See README.md section 1.
set -euo pipefail

SCENE=${1:-../data/scene2}
USE_GPU=${USE_GPU:-0}

if [ ! -d "$SCENE/images" ]; then
  echo "error: $SCENE/images not found" >&2
  exit 1
fi

# --data_type individual: treat the photos as an unordered set and compare all pairs.
# (The 'video' alternative only compares nearby frames, which is faster but assumes
# consecutive images overlap.)
colmap automatic_reconstructor \
  --image_path "$SCENE/images" --workspace_path "$SCENE" \
  --data_type individual --single_camera 1 --camera_model OPENCV \
  --dense 0 --use_gpu "$USE_GPU"

# Remove lens distortion, so the camera becomes a simple pinhole. Training assumes this.
colmap image_undistorter \
  --image_path "$SCENE/images" --input_path "$SCENE/sparse/0" \
  --output_path "$SCENE/undistorted" --output_type COLMAP

# Check "Registered images" against your photo count. Much lower means the reconstruction
# fragmented, and no training setting will fix that.
colmap model_analyzer --path "$SCENE/undistorted/sparse"

# Mandatory: colmap.py reads the text model, not the binary one.
colmap model_converter \
  --input_path "$SCENE/undistorted/sparse" \
  --output_path "$SCENE/undistorted/sparse" --output_type TXT
