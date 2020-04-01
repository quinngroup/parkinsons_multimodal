import torch
import torch.nn as nn
import torch.nn.functional as F
from Pieces.pieces import *

class Encoder(nn.Module):
    
    def __init__(self):
        
        super(Encoder, self).__init__()
        
        # All residual blocks to be used in the encoder
        self.residuals = nn.ModuleList([
        
            Residual(1, 4, 32),
            Residual(256, 32, 512),
            Residual(72, 8, 128),
            Residual(18, 2, 32)
            
        ])
        
        self.quantizers = nn.ModuleList([
        
            VectorQuantizer(512, 32),
            VectorQuantizer(512, 8),
            VectorQuantizer(512, 2)
        
        ])

        
        self.conv_1 = nn.Sequential(
            
            ResidualConv(4, 8),
            ResidualConv(8, 16)
        
        )
        
        self.conv_2 = nn.Sequential(
        
            ResidualConv(16, 32),
            ResidualConv(32, 64)
        
        )
        
        self.conv_3 = nn.Sequential(
        
            ResidualConv(64, 128),
            ResidualConv(128, 256)
            
        )
        
        self.trans_1 = nn.Sequential(
        
            ResidualTrans(32, 16),
            ResidualTrans(16, 8)
        
        )
        
        self.trans_2 = nn.Sequential(
        
            ResidualTrans(8, 4),
            ResidualTrans(4, 2)
        
        )
    
    def forward(self, x):
        
        out = self.residuals[0](x)
        
        top_conv = self.conv_1(out)
        mid_conv = self.conv_2(top_conv)
        bot_conv = self.conv_3(mid_conv) # TODO del?
        
        out = self.residuals[1](bot_conv)
        bot_loss, bot_quant, _, _ = self.quantizers[0](out)
        
        out = self.trans_1(bot_quant)
        out = torch.cat((out, mid_conv), dim=1)
        out = self.residuals[2](out)
        mid_loss, mid_quant, _, _ = self.quantizers[1](out)
        
        out = self.trans_2(mid_quant)
        out = torch.cat((out, top_conv), dim=1)
        out = self.residuals[3](out)
        top_loss, top_quant, _, _ = self.quantizers[2](out)
        
        return [bot_quant, mid_quant, top_quant], bot_loss + mid_loss + top_loss
    
class Decoder(nn.Module):
    
    def __init__(self):
        
        super(Decoder, self).__init__()
        
        self.trans_1 = nn.Sequential(
        
            ResidualTrans(32, 128),
            ResidualTrans(128, 64)
        
        )
        
        self.trans_2 = nn.Sequential(
        
            ResidualTrans(72, 32),
            ResidualTrans(32, 16)
        
        )
        
        self.trans_3 = nn.Sequential(
        
            ResidualTrans(18, 8),
            ResidualTrans(8, 4)
        
        )
        
        self.residual_1 = Residual(4, 1, 32)
        
    def forward(self, x):
        
        out = self.trans_1(x[0])
        out = torch.cat((x[1], out), dim=1)
        out = self.trans_2(out)
        out = torch.cat((x[2], out), dim=1)
        out = self.trans_3(out)
        
        return self.residual_1(out)
    

class VVQVAE(nn.Module):
    
    def __init__(self):
        
        super(VVQVAE, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        
    def forward(self, x):
        
        quantized, loss = self.encoder(x)
        
        return self.decoder(quantized), loss
        
        