import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Module representing a vector quantization layer. Takes input data, reshapes it
from BCDHW to BHDWC, flattens it to (BDHW, C), then looks up the nearest vector in
an embedding for all BDHW vectors.
"""
class VectorQuantizer(nn.Module):

    """
    Constructor to create layer. Uses num_embeddings and embedding_dim to create
    an embedding layer, where embedding_dim (D in reference paper) must match the 
    C of the input data that will be passed in. Beta is equivalent to commitment 
    cost.
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99, epsilon=1e-5):

        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.beta = beta

        # Initializing embedding
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        
        # Initializing Exponential Moving Average pieces
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    """
    Defines forward pass of data through the quantization layer. 
    """
    def forward(self, x):

        # Convert inputs from BCDHW to BDHWC then flattens to (BDHW, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
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
        
        # convert quantized from BDHWC -> BCDHW
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings
    
"""
Single residual layer as parts of ResNets with FixUp initialization.
Read more: https://arxiv.org/abs/1901.09321
"""
class ResidualConv(nn.Module): # TODO Bottleneck version?

    """
    Constructor to build a residual block.
    """
    def __init__(self, in_channels, out_channels):

        super(ResidualConv, self).__init__()

        # Parameters associated with first convolution operation
        self.bias_1a = nn.Parameter(torch.zeros(1))
        self.bias_1b = nn.Parameter(torch.zeros(1))
        
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                                stride=2, padding=1, bias=False)
        
        
        # Parameters associated with second convolution operation
        self.bias_2a = nn.Parameter(torch.zeros(1))
        self.bias_2b = nn.Parameter(torch.zeros(1))
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding = 1, bias=False)
        
        # Other operations
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)
        self.scale = nn.Parameter(torch.ones(1))
        self.relu = nn.ReLU(inplace=True)

    """
    Defines forward pass through the residual block where the
    input is added to output of conv operations.
    """
    def forward(self, x):

        out = self.conv_1(x + self.bias_1a)
        out = self.relu(out + self.bias_1b)
        
        out = self.conv_2(out + self.bias_2a)
        out = out * self.scale + self.bias_2b
        
        x = self.downsample(x + self.bias_1a)
        
        return self.relu(out + x)


class ResidualTrans(nn.Module): # TODO ICNR Initialization
    
    def __init__(self, in_channels, out_channels):
        
        super(ResidualTrans, self).__init__()
        
        self.residual = nn.Sequential(
        
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=1, padding=0)
        )
        
        self.downsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, output_padding=1)
        
    def forward(self, x):
        
        return self.downsample(x) + self.residual(x)
    
class Residual(nn.Module):
    
    def __init__(self, in_channels, out_channels, inter_channels):
        
        super(Residual, self).__init__()
        
        self.residual = nn.Sequential(
        
            nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, out_channels, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True)
            
        )
        
        self.inter = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        
    
    def forward(self, x):
        
        out_1 = self.residual(x)
        out_2 = self.inter(x)
        
        return out_1 + out_2 
    
    
    
    
    
    
    
