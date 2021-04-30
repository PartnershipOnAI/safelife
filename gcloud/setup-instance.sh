#!/bin/sh

for instance in $* ; do
    gc ssh $instance y # nvidia driver
    #  wait for prelimiary apt-get stuff to run
    gc ssh $instance sudo apt-get install mosh
    gc ssh $instance git clone https://github.com/partnershiponai/safelife
    echo y | conda create -n py37 python=3.7
    conda activate py37
    cd safelife ; pip install -r requirements.txt
    echo $CREDENTIAL | wandb login  # or have it in the tree?
    wandb agent $RUN_ID

done
