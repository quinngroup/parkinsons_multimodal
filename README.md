# Multimodal Parkinson's

# TODO

There's a few #Todo's in the code, need to take a look at them

Verify quality of latent vectors by clustering into healthy and PD patients

Incorporate more modalities and ensure data_init and data_info can track multiple images per entry

Add optimizer and latent space size to checkpoints

Pass phenotypic data to the linear layers after convolutions?

# Running Code

All commands should be run from the root github directory. Run `python src/main.py --help` for more information about arguments.

# Data Download

`python src/main.py -d -k cloud_key.json` where cloud_key.json is the access key to a google cloud bucket containing data from PPMI. It is recommended to put this in the root project directory, *but do not add it to github.*

# Training

`python src/main.py --train [ARGS]` where args can be `-b BATCH_SIZE` `-e EPOCHS` `--save_frequency FREQUENCY`. It is recommended to train the network with `nohup python src/main.py --train [ARGS] &` to avoid stopping training due to SSH disconnecting or any other reason.  
