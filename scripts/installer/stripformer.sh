#!/bin/bash

# Install Stripformer and dependencies
# should be run from repo root (scripts/installer/stripformer.sh)

cd bin
git clone https://github.com/pp00704831/Stripformer.git stripformer
cd stripformer
python3 -m venv env

env/bin/pip3 install torch==1.11.0 torchvision==0.12.0 torchaudio opencv-python tqdm pyyaml joblib glog scikit-image tensorboardX albumentations gdown
env/bin/pip3 install -U albumentations[imgaug]
env/bin/pip3 install albumentations==1.1.0

env/bin/gdown --folder 1YcIwqlgWQw_dhy_h0fqZlnKGptq1eVjZ

patch predict_GoPro_test_results.py < ../../scripts/installer/stripformer.patch
