import os
import torch
import torchinfo
from torchinfo import summary

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download, add_path_to_tsv
from Preprocessing.data_loading import get_dataloader
from Models.VAE import VAE
from Models.VAE import _VAE_NN


tsv_path = os.getcwd() + '/data/participants.tsv'
data_info_path = "data_info.csv"

if __name__ == "__main__":

    args = get_args()

    # If data needs to be downloaded
    #if args['download']:
    #    organized_data_download(args['key_path'], args['bucket'])

    #add_path_to_tsv(tsv_path)
    # Converting data to a dataloader
    data = get_dataloader(data_info_path, batch_size=args['batch_size'])

    vae = VAE(args['latent_size'])
    vae_model = _VAE_NN(args['latent_size'])
    
    batch_size = 1
    summary(vae_model, input_size=(batch_size, 1, 193, 229, 193))
    #summary(vae_model, input_size=(64, 1, 3, 3, 3))

    #if args['load_checkpoint'] is not None:
    #    vae.load_checkpoint(args['load_checkpoint'])

    if args['train']:
        vae.train(data, args['epochs'], save_frequency=args['save_frequency'])

    #if vae.num_epochs_completed == 0:
    #    print("[WARNING] Using VAE that has not been trained")

