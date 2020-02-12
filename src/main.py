import os

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download
from Preprocessing.data_loading import get_dataloader
from Models.VAE import VAE

import matplotlib.pyplot as plt

data_info_path = os.getcwd() + '/data_info.csv'


if __name__ == "__main__":

    args = get_args()

    # If data needs to be downloaded
    if args['download']:
        organized_data_download(args['key_path'], args['bucket'])

    # Converting data to a dataloader
    data = get_dataloader(data_info_path, batch_size=args['batch_size'])

    vae = VAE(args['latent_size'])

    if args['train']:
        vae.train(data, 3, log_frequency=1)


