#!/bin/bash

# Install MPRnet and dependencies
# should be run from repo root (scripts/installer/mprnet.sh)

cd bin
git clone https://github.com/swz30/MPRNet.git mprnet
cd mprnet
python3 -m venv env

env/bin/pip3 install torch==1.11.0 torchvision==0.12.0  matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm gdown
env/bin/python3 pytorch-gradual-warmup-lr/setup.py install

env/bin/gdown --id 1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb -O Deblurring/pretrained_models/model_deblurring.pth
env/bin/gdown --id 1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw -O Denoising/pretrained_models/model_denoising.pth

# create run script
cat > mprnet <<- EOM
#!/bin/bash
venv="\$(dirname \$(realpath \$0))"
\${venv}/env/bin/python bin/mprnet/demo.py "\$@"
EOM
chmod +x mprnet
