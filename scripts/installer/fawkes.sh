#!/bin/bash

# Install fawkes and dependencies
# should be run from repo root (scripts/installer/fawkes.sh)
# Your fawkes installation will be available in bin/fawkes/fawkes

# create venv
mkdir -p bin/fawkes
cd bin/fawkes
python3 -m venv env

# install fawkes
git clone https://github.com/Shawn-Shan/fawkes.git git
cd git
git checkout 600fb825689ecafc1f1abce8b0b3bd6fcd27e355
sed -i 's/tensorflow==2.4.1/tensorflow==2.8.2/' setup.py
sed -i 's/keras==2.4.3/keras==2.8.0/' setup.py
sed -i 's/import keras./import tensorflow.keras./' fawkes/*
sed -i 's/from keras./from tensorflow.keras./' fawkes/*
../env/bin/pip install .

# create fawkes binary shortcut
cd ..
ln -s env/bin/fawkes fawkes

# run fawkes once to download model
./fawkes -m high
