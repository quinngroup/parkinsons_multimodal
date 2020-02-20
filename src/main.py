import os

import torch

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download
from Preprocessing.data_loading import get_dataloader
from Models.VAE import VAE

import numpy as np

data_info_path = os.getcwd() + '/data_info.csv'


if __name__ == "__main__":

    args = get_args()

    # If data needs to be downloaded
    if args['download']:
        organized_data_download(args['key_path'], args['bucket'])

    # Converting data to a dataloader
    data = get_dataloader(data_info_path, batch_size=args['batch_size'])

    vae = VAE(args['latent_size'])

    if args['load_checkpoint'] is not None:
        vae.load_checkpoint(args['load_checkpoint'])

    if args['train']:
        vae.train(data, args['epochs'], save_frequency=args['save_frequency'])

    if vae.num_epochs_completed == 0:
        print("[WARNING] Using VAE that has not been trained")

