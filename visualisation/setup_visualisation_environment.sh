#!/usr/bin/env bash

## this section is only for Ubuntu 18.04 machines. TODO add support for more versions.
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9DA31620334BD75D9DCB49F368818C72E52529D4

echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
##

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
git checkout visualise

# install the python packages.
pip3 install --user -e ./datalake
pip3 install --user -e ./utils
pip3 install --user -e ./visualsation

# install mongodb
sudo apt-get install -y mongodb-org

# start mongo
sudo service mongod start
