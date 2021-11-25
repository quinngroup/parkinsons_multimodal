import os
import torch
import torchinfo
from torchinfo import summary

from Utils.argparsing import get_args
from Preprocessing.data_init import organized_data_download, add_path_to_tsv
from Preprocessing.data_loading import get_dataloader
from Models.VAE import VAE
from Models.VAE import _VAE_NN
from Models.clustering import umap_visualization
from Models.clustering import pca_visualization

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

tsv_path = os.getcwd() + '/data/participants.tsv'
data_info_path = "data_info.csv"

if __name__ == "__main__":

    args = get_args()

    if args['data_info']:
        add_path_to_tsv(tsv_path)
    
    # Converting data to dataloaders
    data, test_data = get_dataloader(data_info_path, batch_size=args['batch_size'])

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
       
    pca_visualization(vae, test_data)
    umap_visualization(vae, test_data)
    
    '''vectors_list = []
    for tensor in big_mus:
        for batch in tensor:
            numpy = batch.detach().cpu().numpy()
            vectors_list.append(numpy)
    big_mus_arr = np.array(vectors_list)
        
    # PCA
    pca_embedding = PCA(n_components=2).fit_transform(big_mus_arr)
    labels = [ 0 if x==('PD') else 1 for x in big_labels]
    plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=labels, cmap='Spectral')
    plt.savefig('pca_2_dim.png')
    plt.close()
    '''