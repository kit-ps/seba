#!/bin/bash

# Install ciagan and dependencies
# should be run from repo root (scripts/installer/ciagan.sh)

# create venv
mkdir -p bin/ciagan
cd bin/ciagan
python3 -m venv env

# install ciagan
git clone https://github.com/dvl-tum/ciagan.git git
cd git
git checkout 38d0eac9d03d3970607a59b43f8041dfac05887d
cd ..

env/bin/pip install wheel
env/bin/pip install -r git/requirements.txt
env/bin/pip install opencv-python dlib gdown

# create run script
cat > ciagan <<- EOM
#!/bin/bash
venv="\$(dirname \$(realpath \$0))"
\${venv}/env/bin/python git/source/"\$@"
EOM
chmod +x ciagan

# get landmarks.dat
curl -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# get model
env/bin/gdown --id 1j5iT-SvvbC-JRy7qvY-eEP4sLzvoh8Ut -O model

patch -d git/source < ../../scripts/installer/ciagan.patch
