import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Module that performs two convolution operations. 

Takes input data, converts the number of channels to out_channels, normalizes the batch, and passes through 
ReLU activation function. Does the sample for the second pass, but keeps the output channels constant. Performs
appropriate padding to avoid changing the size of the data.
"""
class DoubleConv(nn.Module):

    """
    Builds the module as a torch sequential neural network that can be used in the VAE.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)

        )

    """
    Defines behavior when taking forward pass of the network.
    """
    def forward(self, x):
        print("Finished a double conv layer")
        return self.double_conv(x)

"""
Module that downsamples input data to reduce dimensionality and computational cost by using a max pooling
layer in 3D.
"""
class Downsample(nn.Module):

    """
    Builds the module as a single layer that can be used in the VAE.
    """
    def __init__(self):
        super().__init__()

        self.downsample = nn.MaxPool3d((2, 2, 2))

    """
    Defines behavior when taking forward pass of the network.
    """
    def forward(self, x):

        return self.downsample(x)

"""
Module that upsamples the input data for use in the decoder portion of the VAE.
"""
# TODO Can use ConvTranspose3d, MaxUnpool3d, Upsample (trilinear), or interpolate
class Upsample(nn.Module):

    """
    Builds the module as a single layer that can be used in the VAE.
    """
    def __init__(self, in_channels):
        super().__init__()
    
        # TODO Divide channels by 2?
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2, 2, 2), stride=2)

    """
    Defines behavior when taking forward pass of the network.
    """
    def forward(self, x):

        return self.up(x)


"""
Module that creates a fully connected layer to be used once on 1D data. Can be used in either the encoder
or decoder portion of the VAE.

Takes input data, passes through linear layer, normalizes data, passes through ReLU, then applies dropout to
help reduce overfitting.
"""
class FC(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        
        return self.fc(x)


def __make_decoder_conv_layer(self, size, in_c, out_c):

        conv_layer = nn.Sequential(

            nn.Upsample(size=size, mode='trilinear'),
            nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=(1, 1, 1)),
            nn.LeakyReLU()
        )
        return conv_layer


