#!/bin/bash

# Install arcface and dependencies
# should be run from repo root (scripts/installer/arcface.sh)

mkdir -p bin/
cd bin

# download arcface (use svn to not download entire repo...)
git clone  -n --depth=1 --filter=tree:0 https://github.com/deepinsight/insightface arcface
cd arcface
git sparse-checkout set --no-cone recognition/arcface_torch
mv recognition/arcface_torch/* .
rm -r recognition

# install requirements
sed -i 's/sklearn/scikit-learn/' requirements.txt
pip3 install -r requirement.txt
