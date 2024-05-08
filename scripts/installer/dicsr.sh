#!/bin/bash

# Install DIC-SR and dependencies
# should be run from repo root (scripts/installer/dicsr.sh)
# Your DIC-SR installation will be available in bin/dicsr/dicsr

# create venv
cd bin
git clone https://github.com/Maclory/Deep-Iterative-Collaboration dicsr
cd dicsr

python3 -m venv env
env/bin/pip install torch numpy opencv-python tqdm imageio pandas matplotlib tensorboardX torchvision

# create run script
cat > dicsr <<- EOM
#!/bin/bash
venv="\$(dirname \$(realpath \$0))"
\${venv}/env/bin/python \${venv}/code/test.py "\$@"
EOM
chmod +x dicsr
