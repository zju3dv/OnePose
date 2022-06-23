#!/bin/bash
PROJECT_DIR="$(pwd)"
VIDEO_PATH=$1

echo '-------------------'
echo 'Parse full image: '
echo '-------------------'

# Parse full image from Frames.m4v
python $PROJECT_DIR/video2img.py \
    --video_file ${VIDEO_PATH}
    