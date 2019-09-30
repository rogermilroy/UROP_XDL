#!/usr/bin/env bash

# Note. must be run as root or su.
# update apt and install among other things the "latest" python currently 3.6.7.
sudo apt update
sudo apt install -y software-properties-common

# install pip.
sudo apt install -y python3-pip

# install git
sudo apt install -y git

# clone the project and download dependencies
mkdir code
cd code
git clone https://github.com/rogermilroy/UROP_XDL.git

cd UROP_XDL

# switch to current development branch
git checkout visualise

pip3 install --user -e ./extraction
pip3 install --user -e ./utils
pip3 install --user -e ./testing