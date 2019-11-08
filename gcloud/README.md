# Running SafeLife in Google Cloud

This directory contains a number of scripts that make it a little bit easier to start remote training jobs on Google Cloud and download their outputs. The following instructions should help you get up and running.

Note that there are many cloud computing platforms which may perform equally well. These scripts are just a convenience for those that happen to be using *gcloud*.

## Signing up and creating new instances

The first thing to do is download the [gcloud command-line tool](https://cloud.google.com/sdk/) and sign up for a Google Cloud account. Google has reasonably good documentation on how to do this. If you wish to run on GPU-accelerated machines, you will need to [request a non-zero quota of simultaneous GPUs](https://cloud.google.com/compute/quotas). Lists of available GPUs and the regions that they are available in [can be found here](https://cloud.google.com/compute/docs/gpus/).

In order to actually run any training, you will need to provision virtual machines on which to run. Instances can easily be reused. They come with persistent disks, so you can stop and start them and retain all of your training data. To provision a new virtual machine, or *instance*, run e.g.

    gcloud compute instances create $INSTANCE --image-family tf-1-14-cu100 --image-project deeplearning-platform-release --boot-disk-size=200GB  --accelerator type=nvidia-tesla-k80 --maintenance-policy TERMINATE --restart-on-failure --metadata startup-script="~/current_job/src/start-training --shutdown"

The startup script ensures that the instances will restart the last active job if it needs to be restarted.

*(To do: check that the startup script is actually working correctly.)*


## Starting new jobs

Jobs run on gcloud instances, so first make sure that your instance is running. Use `gcloud compute instances list` to view the status of all of your instances, and `gcloud compute instances start $INSTANCE` to start a particular instance. Once the instance is up and running you can start a job using e.g.

    gcloud/start-remote-job $INSTANCE $JOB_NAME --port=6006

That will copy over the benchmarks directory into the appropriately named folder and set up port forwarding for tensorboard onto the specified local port. It will then run `./start-training $JOB_NAME` on the remote instance inside of a `tmux` session. To (gracefully) close the connection, hit `ctrl-b, d`.

Once a job is running, the following directories will be created on the remote instance:

    ~/JOB_NAME/       [copy of the root project folder]
    ~/JOB_NAME/data/  [all the training data gets stored here]
    ~/current_job     [link to whichever job has most recently been run]

Note that there is no per-job directory for dependencies or python virtual environments. It's assumed that each instance is primarily just running SafeLife, and that the dependencies are fairly static between training runs and code updates, so the installed dependencies are global for all jobs on a given instance.


## Other handy things to do

- Start a remote notebook:

        gcloud/ssh $INSTANCE -L 8887:localhost:8888 jupyter notebook

  Then you should be able to visit the notebook at `https://localhost:8887/?token=xxx`. The token should be printed during notebook startup.

- Sync job back to localhost:

        gcloud/rsync -r $INSTANCE:~/current_job/ ./jobs
  or

        gcloud/sync-current $INSTANCE

  This will copy the entire job directory on the remote machine into a local "jobs" folder.

- Reconnect to a running remote job:

        gcloud/resume $INSTANCE
