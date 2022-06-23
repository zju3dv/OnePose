#!/bin/bash
REPO_ROOT=$(pwd)
echo "work space:$REPO_ROOT"

# Download code and pretrained model of SuperPoint:
mkdir -p $REPO_ROOT/data/models/extractors/SuperPoint
# cd $REPO_ROOT/src/models/extractors/SuperPoint
# wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/superpoint.py
cd $REPO_ROOT/data/models/extractors/SuperPoint
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth

# Download code and pretrained model of SuperGlue:
mkdir -p $REPO_ROOT/data/models/matchers/SuperGlue
# cd $REPO_ROOT/src/models/matchers/SuperGlue
# wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/superglue.py
cd $REPO_ROOT/data/models/matchers/SuperGlue
wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth