# Multimodal Parkinson's

# TODO

There's a few #Todo's in the code, need to take a look at them

Verify quality of latent vectors by clustering into healthy and PD patients

Incorporate more modalities and ensure data_init and data_info can track multiple images per entry

Convert to beta VAE learning Bernoulli?

Implement Spatial Broadcast Decoding instead of conv transpose

# Running Code

All commands should be run from the root github directory. Run `python src/main.py --help` for more information about arguments.

# Training

`python src/main.py --train [ARGS]` where args can be `-b BATCH_SIZE` `-e EPOCHS` `--save_frequency FREQUENCY`. It is recommended to train the network with `nohup python src/main.py --train [ARGS] &` to avoid stopping training due to SSH disconnecting or any other reason.  
