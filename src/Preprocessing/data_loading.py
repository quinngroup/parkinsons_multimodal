import torch
import warnings
from torch.utils.data import Dataset, DataLoader, Subset
from Utils.niiparsing import load_nii
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split

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
        self.targets = self.data_info_df['Group'].tolist()

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
        DEPTH = 193
        HEIGHT = 229
        WIDTH = 193

        if torch.is_tensor(idx):
            idx = idx.tolist()

        images_df_row = self.data_info_df.iloc[idx]

        # Loading image
        image = torch.tensor(load_nii(images_df_row['Path']))
        #image = image.permute(2, 0, 1)

        image_size = image.size()
        #print(images_df_row['Subject_Num'], ' ', image_size)

        # Trilinear interpolation to convert all images to the same size.
        # Due to torch requirements, the input data to the interpolation function
        # will be 5D in the NCDHW format.
        # Should not be needed now as they should all be the same size... 193/229/193.
        #image = F.interpolate(
        #    image.view(1, 1, image_size[0], image_size[1], image_size[2]),
        #    size=(DEPTH, HEIGHT, WIDTH),
        #    mode='trilinear')

        return {

            'image' : image.view(CHANNELS, DEPTH, HEIGHT, WIDTH),
            'modality' : images_df_row['Modality'],
            'description' : images_df_row['Description'],
            'subject_num' : images_df_row['Subject_Num'],
            'subject_id' : images_df_row['Subject'],
            'image_id' : images_df_row['Image Data ID'],
            'group' : images_df_row['Group'],
            'orig_size' : image_size,
            'idx' : idx

        }

"""
Creates a torch DataLoader that loads data into memory more efficiently and handles batching.
Returns a train_loader and test_loader with an 80/20 train/test split.
"""
def get_dataloader(data_info_path, batch_size=64, shuffle=True, num_workers=4):
    full_dataset = __ParkinsonsDataset(data_info_path)
    train_indices, test_indices, _, _ = train_test_split(range(len(full_dataset)), full_dataset.targets, stratify=full_dataset.targets, test_size=0.2)
    
    # generate subset based on indices
    train_split = Subset(full_dataset, train_indices)
    test_split = Subset(full_dataset, test_indices)

    train_loader = DataLoader(dataset=train_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_split)

    return train_loader, test_loader
