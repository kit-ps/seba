#!/bin/bash

# Install arcface and dependencies
# should be run from repo root (scripts/installer/arcface.sh)

mkdir -p bin/
cd bin

# download arcface (use svn to not download entire repo...)
svn export https://github.com/deepinsight/insightface/trunk/recognition/arcface_torch arcface
cd arcface

# install requirements
pip3 install -r requirement.txt
