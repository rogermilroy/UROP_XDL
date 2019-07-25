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

# switch to current development branch
cd UROP_XDL
# switch to current development branch
git checkout implement-datalake

cd datalake

pip3 install --user -e .

pip3 install --user -e ../utils


# TODO install MongoDB and set up.