import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

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
class Upsample(nn.Module):

    """
    Builds the module as a single layer that can be used in the VAE.
    """
    def __init__(self, in_channels):
        super().__init__()
    
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2, 2, 2), stride=2)

    """
    Defines behavior when taking forward pass of the network.
    """
    def forward(self, x):

        return self.up(x)

"""
Add later
"""
class OutputUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()        
        self.output_conv = nn.ConvTranspose3d(in_channels, 1, kernel_size=(5, 5, 5), stride=(6, 7, 6), padding=(1,0,1), dilation=2, output_padding=(0,3,0))
        
    def forward(self, x):
        return self.output_conv(x)

    
"""
Spatial broadcast decoder.
"""
class SpatialBroadcastDecoder(nn.Module):
    '''
    Constructs spatial broadcast decoder
    Adapted from https://github.com/quinngroup/CiliaRepresentation/blob/e1f80399818c2edc7a788a3194da42d9c5cd3e8f/VTP/utils/nn.py#L78-L105
    @param input_length width of image
    @param device torch device for computations
    @param lsdim dimensionality of latent space
    @param kernel_size size of size-preserving kernel. Must be odd.
    @param channels list of output-channels for each of the four size-preserving convolutional layers
    '''
    def __init__(self):
        super().__init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64,1])
        self.input_length=input_length
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        #Size-Preserving Convolutions
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(channels[3], channels[4], 1)

   # def forward(self, z):
   #     print(z.shape)
   #     return z
    '''
    Applies the spatial broadcast decoder to a code z
    @param z the code to be decoded
    @return the decoding of z
    '''
    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_length,self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)

        xAxisVector = stepTensor.view(1,1,self.input_length,1)
        yAxisVector = stepTensor.view(1,1,1,self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)

        base = torch.cat((xPlane, yPlane, base), 1)

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x
  

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
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15)

        )

    def forward(self, x):

        return self.fc(x)
