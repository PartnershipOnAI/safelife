#!/bin/sh

name="$1"
zone="$2"

PROJECT=`cat gcloud/.project-id`
gcloud compute instances create "$name" --image-family pytorch-latest-gpu --image-project deeplearning-platform-release --boot-disk-size=200GB  --accelerator type=nvidia-tesla-t4 --machine-type=n1-standard-2 --maintenance-policy TERMINATE --billing-project $PROJECT --preemptible --zone "$zone"

echo "y
> sudo apt-get install mosh" | gcloud/ssh "$name"
