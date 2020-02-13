# Multimodal Parkinson's

# TODO

There's a few #Todo's in the code, need to take a look at them

Change normalizing all the input data to normalizing data when batching. This seemed to create issues with the loss function though when removing the initial normalization

Make network deeper

Train/Save/Tweak VAE (need to find time slot when hardware is available)

Verify quality of latent vectors by clustering into healthy and PD patients

Incorporate more modalities and ensure data_init and data_info can track multiple images per entry

Add optimizer and latent space size to checkpoints

All commands should be run from the root github directory. Run `python src/main.py --help` for more information about arguments.

# Data Download

`python src/main.py -d -k cloud_key.json` where cloud_key.json is the access key to a google cloud bucket containing data from PPMI. It is recommended to put this in the root project directory, *but do not add it to github.*
