import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from Utils.niiparsing import load_nii
import torch.nn.functional as F

import pandas as pd

warnings.filterwarnings("ignore")

"""
Represents the entire PPMI dataset that is downloaded. Is manipulated by the 
torch DataLoader to extract batches of data.
"""
class __ParkinsonsDataset(Dataset):

    """
    Constructor that needs to have access to data_info which contains paths
    to all data.
    """
    def __init__(self, data_info_path):

        self.data_info_df = pd.read_csv(data_info_path)

    """
    Number of samples that have been downloaded.
    """
    def __len__(self):

        return len(self.data_info_df)

    """
    Gets a single sample from the dataset. Returns dictionary of metadata about the
    sample for later processing. Image itself is returned as a tensor.
    """
    def __getitem__(self, idx):

        # Common size that all images will be converted to
        # Formatted to follow torch's NCDHW where N will be handled by the DataLoader
        CHANNELS = 1
        DEPTH = 128
        HEIGHT = 128
        WIDTH = 128

        if torch.is_tensor(idx):
            idx = idx.tolist()

        images_df_row = self.data_info_df.iloc[idx]

        # Loading image
        image = torch.tensor(load_nii(images_df_row['Path']))
        image = image.permute(2, 0, 1)

        image_size = image.size()

        # Trilinear interpolation to convert all images to the same size.
        # Due to torch requirements, the input data to the interpolation function
        # will be 5D in the NCDHW format.
        # TODO change to align_corvers=True?
        image = F.interpolate(
            image.view(1, 1, image_size[0], image_size[1], image_size[2]),
            size=(DEPTH, HEIGHT, WIDTH),
            mode='trilinear')

        return {

            #'image' : torch.nn.functional.normalize(image.view(CHANNELS, DEPTH, HEIGHT, WIDTH), dim=0),
            'image' : image.view(CHANNELS, DEPTH, HEIGHT, WIDTH),
            'modality' : images_df_row['Modality'],
            'description' : images_df_row['Description'],
            'subject_id' : images_df_row['Subject'],
            'image_id' : images_df_row['Image Data ID'],
            'group' : images_df_row['Group'],
            'orig_size' : image_size,
            'idx' : idx

        }

"""
Creates a torch DataLoader that loads data into memory more efficiently and handles batching.
"""
def get_dataloader(data_info_path, batch_size=64, shuffle=True, num_workers=4):

    dataset = __ParkinsonsDataset(data_info_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
