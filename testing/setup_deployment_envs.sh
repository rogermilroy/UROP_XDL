#!/usr/bin/env bash

scp ../extraction/setup_extractor_environment.sh trainingvm:~/setup_extractor_environment.sh

ssh -t trainingvm 'chmod +x setup_extractor_environment.sh; ./setup_extractor_environment.sh'

scp ../datalake/setup_datalake_environment.sh datalakevm:~/setup_datalake_environment.sh

ssh -t datalakevm 'chmod +x setup_datalake_environment.sh; ./setup_datalake_environment.sh'
