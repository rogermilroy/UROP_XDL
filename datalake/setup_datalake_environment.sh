#!/usr/bin/env bash

# Note. must be run as root or su.
# update apt and install among other things the "latest" python currently 3.6.7.
apt update
apt install -y software-properties-common

# install pip.
apt install -y python3-pip

# install git
apt install -y git

# clone the project and download dependencies
mkdir code
cd code
git clone https://github.com/rogermilroy/UROP_XDL.git

# switch to current development branch
git checkout implement_datalake

cd UROP_XDL/datalake

pip3 install -r requirements.txt

# TODO install MongoDB and set up.