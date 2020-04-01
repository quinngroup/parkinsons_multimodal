
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import sys


"""
Module representing a vector quantization layer. Takes input data, reshapes it
from BCHW to BHWC, flattens it to (BHW, C), then looks up the nearest vector in
an embedding for al BHW vectors.
"""
class VectorQuantizer(nn.Module):

    """
    Constructor to create layer. Uses num_embeddings and embedding_dim to create
    and embedding layer, where embedding_dim (D in reference paper) must match the 
    C of the input data that will be passed in. Beta is equivalent to commitment 
    cost.
    """
    def __init__(self, num_embeddings, embedding_dim, beta, decay=0.99, epsilon=1e-5):

        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.beta = beta

        # Initializing embedding
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    """
    Defines forward pass of data through the quantization layer. 
    """
    def forward(self, x):

        # Convert inputs from BCHW to BHWC then flattens to (BHW, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        input_shape = x.shape
        flat_x = x.view(-1, self._embedding_dim)

        # Calculate distances from each vector in the flattened input to vectors
        # in the embedding according to ||z - e||2 Where z is vector of input data
        # and e is a vector in the embedding. Output will be (BHW, D).
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self._embedding.weight.t()))

        # Take argmin to find the closest matching vector in embedding to the input
        # vectors. This give the indices of the vectors from the embedding to be
        # used which will be quantized and unflattened.
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_x)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        loss = self.beta * e_latent_loss
        
        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class SubPixelConvolution(nn.Module):

    def __init__(self, upscale_factor):
        
        super(SubPixelConvolution, self).__init__()

        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.conv = nn.Conv2d(3, 3, 1, stride=upscale_factor)

    def forward(self, x):

        # print("SubPixlConvolution intial size: ", x.size())       
 
        x = self.conv(self.shuffle(x))

        # print("Post size: ", x.size())

        return x


"""
Single residual layer as parts of ResNets. Each block is implemented as
A 2D convolution operation, relu, 2D convolution operation, relu, followed
by a 2D convolution operation to match output channels. The output of this 
is then added to the input.
"""
class ResidualBlock(nn.Module):

    """
    Constructor to build a residual block.
    """
    def __init__(self, in_channels, residual_hiddens=32):

        super(ResidualBlock, self).__init__()

        # TODO Batch norm?
        self.residual_block = nn.Sequential(

            nn.ReLU(True),
            
            nn.Conv2d(in_channels, residual_hiddens,
                kernel_size=3, padding=1),
            
            nn.ReLU(True),
            
            nn.Conv2d(residual_hiddens, in_channels,
                kernel_size=1)
   
        )

    """
    Defines forward pass through the residual block where the
    input is added to output of conv operations.
    """
    def forward(self, x):

        return x + self.residual_block(x)


class BottomConvBlock(nn.Module):

  """
  Encoder constructor.
  """
  def __init__(self):

    super(BottomConvBlock, self).__init__()

    self.convs = nn.Sequential(

      nn.Conv2d(3, 64, 4, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(64, 128, 4, stride=2, padding=1),
      nn.ReLU(True),
      nn.Conv2d(128, 128, 3, padding=1),

    )

    self.residual_stack = nn.Sequential(

      ResidualBlock(128),
      ResidualBlock(128),
      nn.ReLU(True)

    )

  """
  Defines forward pass through the encoder.
  """
  def forward(self, x):

    x = self.convs(x)
    return self.residual_stack(x)


"""
Encoder pipeline of the VQ-VAE. Contains the convolutional 
residual blocks.
"""
class MiddleConvBlock(nn.Module):

    """
    Encoder constructor.
    """
    def __init__(self):

        super(MiddleConvBlock, self).__init__()

        self.convs = nn.Sequential(
        
            nn.Conv2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1),
        
        )
        
        self.residual_stack = nn.Sequential(
        
            ResidualBlock(128),
            ResidualBlock(128),
            nn.ReLU(True)
            
        )

    """
    Defines forward pass through the encoder.
    """
    def forward(self, x):

        x = self.convs(x)
        return self.residual_stack(x)


"""
Decoder pipeline of the VQ-VAE. Contains the deconvolutional 
residual blocks
"""
class BottomTransposeBlock(nn.Module):

    """
    Decoder constructor.
    """
    def __init__(self, total_embedding_dim):

        super(BottomTransposeBlock, self).__init__()
        
        self.conv = nn.Conv2d(total_embedding_dim, 128, 3, padding=1)

        self.residual_stack = nn.Sequential(

            ResidualBlock(128),
            ResidualBlock(128),
            nn.ReLU(True)

        )
        
        self.conv_trans = nn.Sequential(
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 12, 4, stride=2, padding=1)
            
        )

    """
    Defines forward pass through the decoder.
    """
    def forward(self, x):
        
        x = self.conv(x)
        x = self.residual_stack(x)
        return self.conv_trans(x)


"""
Decoder pipeline of the VQ-VAE. Contains the deconvolutional 
residual blocks
"""
class MiddleTransposeBlock(nn.Module):

    """
    Decoder constructor.
    """
    def __init__(self, middle_embedding_dim, bottom_embedding_dim):

        super(MiddleTransposeBlock, self).__init__()
        
        self.conv = nn.Conv2d(middle_embedding_dim, 128, 3, padding=1)

        self.residual_stack = nn.Sequential(


            ResidualBlock(128),
            ResidualBlock(128),
            nn.ReLU(True)

        )
        
        # TODO 128 here or bottom embedding dim?
        self.conv_trans = nn.ConvTranspose2d(128, bottom_embedding_dim, 4, stride=2, padding=1)

    """
    Defines forward pass through the decoder.
    """
    def forward(self, x):
        
        x = self.conv(x)
        x = self.residual_stack(x)
        return self.conv_trans(x)


class Encoder(nn.Module):
    
    def __init__(self, 
                 bottom_num_embeddings, bottom_embedding_dim,
                 middle_num_embeddings, middle_embedding_dim,
                 beta):

        super(Encoder, self).__init__()
        
        self.bottom_conv = BottomConvBlock()
        self.middle_conv = MiddleConvBlock()

        self.bottom_quantizer = VectorQuantizer(bottom_num_embeddings, bottom_embedding_dim, beta)
        self.middle_quantizer = VectorQuantizer(middle_num_embeddings, middle_embedding_dim, beta)
        
        self.middle_transpose = MiddleTransposeBlock(middle_embedding_dim, bottom_embedding_dim)

        self.intermediate_middle_conv = nn.Conv2d(128, middle_embedding_dim, 1)
        self.intermediate_bottom_conv = nn.Conv2d(bottom_embedding_dim + 128, bottom_embedding_dim, 1)

    def forward(self, x):

        conved_bottom = self.bottom_conv(x) # [B, 128, 8, 8]

        conved_middle = self.middle_conv(conved_bottom) # [B, 128, 4, 4]
        conved_middle = self.intermediate_middle_conv(conved_middle) # [B, 64, 4, 4]

        quantized_middle_loss, quantized_middle, perplexity_middle, _ = self.middle_quantizer(conved_middle) # [B, 64, 4, 4]

        transposed_middle = self.middle_transpose(quantized_middle) # [B, 128, 4, 4]
        transposed_middle = torch.cat([transposed_middle, conved_bottom], 1) # 192
        transposed_middle = self.intermediate_bottom_conv(transposed_middle)

        quantized_bottom_loss, quantized_bottom, perplexity_bottom, _ = self.bottom_quantizer(transposed_middle)

        return quantized_middle_loss + quantized_bottom_loss, quantized_bottom, quantized_middle, perplexity_middle + perplexity_bottom


class Decoder(nn.Module):

    def __init__(self, bottom_embedding_dim, middle_embedding_dim, upscale_factor=2):

        super(Decoder, self).__init__()

        self.bottom_transpose = BottomTransposeBlock(bottom_embedding_dim + middle_embedding_dim)
        self.subpixel_conv = SubPixelConvolution(upscale_factor)

    def forward(self, x):

        x = self.bottom_transpose(x)
        return self.subpixel_conv(x)


"""
Full VQ-VAE model that combines the flow of input data through the
encoder, quantization block, and decoder.
"""
class VVQModel(nn.Module):

    """
    VQ-VAE constructor that takes parameters for the quantization block.
    """
    def __init__(self, 
                 bottom_num_embeddings, bottom_embedding_dim,
                 middle_num_embeddings, middle_embedding_dim,
                 beta):

        super(VVQModel, self).__init__()

        self.encoder = Encoder(
            
            bottom_num_embeddings=bottom_num_embeddings, 
            bottom_embedding_dim=bottom_embedding_dim,
            middle_num_embeddings=middle_num_embeddings,
            middle_embedding_dim=middle_embedding_dim,
            beta = beta
        
        )
        
        self.decoder = Decoder(bottom_embedding_dim, middle_embedding_dim)
        self.intermediate_transpose = nn.ConvTranspose2d(middle_embedding_dim, middle_embedding_dim, 4, stride=2, padding=1)

    """
    Defines forward pass through the VQ-VAE and returns
    necessary loss metrics.
    """
    def forward(self, x):
        
        total_quantized_loss, quantized_bottom, quantized_middle, total_perplexity = self.encoder(x)

        quantized_middle = self.intermediate_transpose(quantized_middle)
        quantized = torch.cat([quantized_middle, quantized_bottom], 1)
        
        reconstructed_x = self.decoder(quantized)

        return total_quantized_loss, reconstructed_x, total_perplexity
