#!/usr/bin/env bash

scp ././../extraction/setup_extractor_environment.sh trainingvm:~/setup_extractor_environment.sh

ssh -t trainingvm 'chmod +x setup_extractor_environment.sh; sudo ./setup_extractor_environment.sh'
