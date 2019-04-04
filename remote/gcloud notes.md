# Creating new instances

This is pretty straightforward. I want to keep a few different instances on hand and reuse them. The instances should be shutdown between use, but I should only need to do the whole creation and installation once.

    gcloud compute instances create $INSTANCE --image-family tf-1-13-gpu --image-project ml-images --boot-disk-size=200GB  --accelerator type=nvidia-tesla-k80 --maintenance-policy TERMINATE --restart-on-failure --metadata startup-script="~/current_run/start-job --shutdown"

where `$INSTANCE` should be of the form `safety-gpu-n`. Note that we needed to request a quota increase to access Google's GPU (default quota is zero).


# Instance organization

The instances will be devoted to this project, so I'm not going to worry about setting up virtual environments to prevent dependency conflicts. The file structure will look like this:

    ~/2019-01-01-RUN_NAME/  [copy of the safety-benchmark folder]
    ~/2019-01-01-RUN_NAME/data/active_run.txt  [to recover from crashes]
    ~/2019-01-01-RUN_NAME/data/RUN_NAME/  [the logdir for each run]
    ~/current_run  [link to whatever is currently running]

There is very little cost to copying the entirety of the code each time we want to do a new run, and it makes it easy to compare (e.g., via a diff), what changed between subsequent runs.


# Starting new runs

To start a new run, just execute

    remote/gcloud-start-run $INSTANCE $RUN_NAME --port=6006

That will copy over the benchmarks directory into the appropriately named folder and set up port forwarding for tensorboard onto the specified local port. It will then run `./start-job $RUN_NAME` on the remote instance. Hopefully nothing in this folder will actually need to be changed, and all changes to the parameters of the run itself should happen in `start-job`. Probably shouldn't be too strict about this though.


# Other handy things to do

- Start a remote notebook:

        gcloud compute ssh $INSTANCE -- -L 8887:localhost:8888 jupyter notebook

  Then you should be able to visit the notebook at `https://localhost:8887/?token=xxx`.

- Sync data back to localhost:

        remote/gcloud-rsync -r $INSTANCE:~/current_run/data/ ./data
