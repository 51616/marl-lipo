#!/bin/bash
sudo apt-get install gifsicle xvfb -y
conda install -c conda-forge cudatoolkit=11.1 cudnn=8.4.1
pip install --upgrade pip
pip install setuptools==66
pip install wheel==0.38.4
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install protobuf==3.20

# install rebar
cd coop_marl/utils/
git clone https://github.com/51616/rebar.git
cd rebar
python setup.py develop
cd ../../..

cd coop_marl/envs/overcooked/
pip install -e .
cd ../../..

# install coop_marl
python setup.py develop
