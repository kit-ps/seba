#!/bin/bash

# Install Pix2Pix and dependencies
# should be run from repo root (scripts/installer/pix2pix.sh)
# Your Pix2Pix installation will be available in bin/pix2pix

# create venv
cd bin
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pix2pix
cd pix2pix

python3 -m venv env
env/bin/pip install -r requirements.txt
env/bin/pip install opencv-python
