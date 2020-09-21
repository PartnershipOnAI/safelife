#!/bin/bash

#PATH=/usr/local/cuda/bin:/opt/anaconda3/bin/:/opt/conda/bin:/opt/conda/condabin:$PATH
source /etc/profile
python3 -m pip install virtualenv
python3 -m virtualenv ~/safelife-venv --system-site-packages

