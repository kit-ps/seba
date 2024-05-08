#!/bin/bash

patch -ld $CONDA_PREFIX/lib/python3.10/site-packages/deepface/ -p0 < scripts/installer/deepface_dlib_0065.patch
patch -ld $CONDA_PREFIX/lib/python3.10/site-packages/deepface/ -p0 < scripts/installer/deepface_normalization.patch
patch -ld $CONDA_PREFIX/lib/python3.10/site-packages/deepface/ -p0 < scripts/installer/deepface_threshold.patch

