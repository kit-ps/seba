#!/bin/bash

# Install DeepPrivacy and dependencies
# should be run from repo root (scripts/installer/deepprivacy.sh)
# Your DeepPrivacy installation will be available in bin/deepprivacy/deepprivacy

# create venv
mkdir -p bin/deepprivacy
cd bin/deepprivacy
python3 -m venv env

# install DeepPrivacy
git clone https://github.com/hukkelas/DeepPrivacy/ git
env/bin/pip install -r git/requirements.txt
env/bin/pip install -e git

# create run script
cat > deepprivacy <<- EOM
#!/bin/bash
venv="\$(dirname \$(realpath \$0))"
\${venv}/env/bin/python -m deep_privacy.cli "\$@"
EOM
chmod +x deepprivacy
