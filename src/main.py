import os
import torch
import torchinfo
from torchinfo import summary

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download, add_path_to_tsv
from Preprocessing.data_loading import get_dataloader
from Models.VAE import VAE
from Models.VAE import _VAE_NN

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

tsv_path = os.getcwd() + '/data/participants.tsv'
data_info_path = "data_info.csv"

if __name__ == "__main__":

    args = get_args()

    if args['data_info']:
        add_path_to_tsv(tsv_path)
    
    # Converting data to a dataloader
    data = get_dataloader(data_info_path, batch_size=args['batch_size'])

    vae = VAE(args['latent_size'])
    vae_model = _VAE_NN(args['latent_size'])
    
    batch_size = 1
    summary(vae_model, input_size=(batch_size, 1, 193, 229, 193))

    if args['load_checkpoint'] is not None:
        vae.load_checkpoint(args['load_checkpoint'])

    if args['train']:
        big_mus, big_labels = vae.train(data, args['epochs'], save_frequency=args['save_frequency'])

    if vae.num_epochs_completed == 0:
        print("[WARNING] Using VAE that has not been trained")
    
    print(len(big_mus), big_mus[0].shape)
    print(len(big_labels), len(big_labels[0]))
    
    print(big_mus[0])
    
    # PCA
    pca_embedding = PCA(n_components=2).fit_transform(big_mus)
    Y_labels = [ 0 if x==('PD') else 1 for x in Y_train]
    plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=Y_labels, cmap='Spectral');