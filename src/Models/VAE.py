import torch
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim
from torch.nn import functional as F
from Models.VAE_pieces import *
import sys, os

""" 
Class that represents the neural network for a Variational Autoencoder (VAE). Learns the underlying
distribution of input data. Instances of this class should only be created by the VAE class as it
acts as a controller for algorithm itself.
"""

class _VAE_NN(nn.Module):

	
    """ 
    Constructor to initialize the VAE.
    """
    def __init__(self, latent_size):

        super(_VAE_NN, self).__init__()

        self.encoder_convolutions = nn.Sequential(

            DoubleConv(1, 64),
            Downsample(),
            DoubleConv(64, 128),
            Downsample(),
            DoubleConv(128, 256),
            Downsample(),
            DoubleConv(256, 512),
            Downsample(),
            Downsample(),
            DoubleConv(512, 1024),
            Downsample(),
            Downsample() # TODO come up with cleaner way of having (1, 1, 1)?

        )

        self.encoder_linear = nn.Sequential(

            FC(1024, 512)

        )

        self.fc_mu = FC(512, latent_size)
        self.fc_logvar = FC(512, latent_size)

        self.decoder_linear = nn.Sequential(

            FC(latent_size, 512),
            FC(512, 1024),

        )

        self.decoder_convolutions = nn.Sequential(
       
           Upsample(1024),
           Upsample(1024),
           DoubleConv(1024, 512),
           Upsample(512),
           Upsample(512),
           DoubleConv(512, 256),
           Upsample(256),
           DoubleConv(256, 128),
           Upsample(128),
           DoubleConv(128, 64),
           Upsample(64),
           DoubleConv(64, 1)

        )

    """
    Passes input data through the encoder portion of the network.
    Returns the mu and log(variance) of the distribution of the input data.
    """
    def encode(self, x):

        x = self.encoder_convolutions(x)

        x = x.view(x.size()[0], -1)

        x = self.encoder_linear(x)

        mu, logvar = self.fc_mu(x), self.fc_logvar(x)

        return mu, logvar

    """
    Decodes a sample (z-vector) from the distribution learned by the encoder.
    """
    def decode(self, z):

        y = self.decoder_linear(z)

        y = y.view(y.size()[0], -1, 1, 1, 1)

        y = self.decoder_convolutions(y)

        return y

    """
    Encodes image, takes a sample from learned distribution, then decodes it.
    """
    def forward(self, x):

        # Performing reshaping here instead of before feeding data to function
        # mu, logvar = self.encode(x.view(-1, self.input_size))
        mu, logvar = self.encode(x)
        z = sample_distribution(mu, logvar)

        return self.decode(z), mu, logvar
"""
Controller for the VAE itself. Handles training, testing, encoding, and decoding.
"""
class VAE():
	
    """
    Handles initializing the VAE and determines correct device to be run on
    """
    def __init__(self, latent_size):

        self.latent_size = latent_size

        if torch.cuda.is_available():
            self.__device = torch.device("cuda")
            self.__model = _VAE_NN(latent_size).cuda()
            self.__model = nn.DataParallel(self.__model)

        else:
            self.__device = torch.device("cpu")

        # self.__model = DDP(_VAE_NN(latent_size).double().to(self.__device))

        self.num_epochs_completed = 0

        print("[INFO] Device detected: %s" % self.__device)

    """
    Calculates the loss function of the VAE encoding and decoding input. Loss is 
    given by calculating the mean squared error loss between the reconstructed input
    and actual input, then adding it to the KL Divergence which measures differences in the 
    distribution of data.
    """
    def __loss_function(self, recon_x, x, mu, logvar):

        # TODO Rethink this
        MSE = F.mse_loss(recon_x, x, reduction='mean')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    """
    Handles training the VAE. Requires a DataLoader which has been set to load only the
    training dataset.
    """
    def train(self, train_loader, epochs, save_frequency, lr=1e-3):

        self.__model.train()
        optimizer = optim.Adam(self.__model.parameters(), lr=lr)

        print("[INFO] Beginning VAE training")

        # Training for selected number of epochs
        for epoch in range(1, epochs + 1):

            train_loss = 0

            # Looping through data batches from the loader
            for batch_idx, batch_data in enumerate(train_loader):

                batch = batch_data['image']

                batch = batch.to(self.__device, dtype=torch.float)
                optimizer.zero_grad()

                reconstructed_batch, mu, logvar = self.__model(batch)

                # Calculating and backpropogating error through the model
                loss = self.__loss_function(reconstructed_batch, batch, mu, logvar)
                loss.backward()
                train_loss += loss.item()

                # Changing model weights to minimize loss
                optimizer.step()

                # Logging
                sys.stdout.write('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(batch)))
                sys.stdout.flush()

            epoch_train_loss = train_loss / len(train_loader.dataset)

            # Logging
            print('\r====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_train_loss) + " " * 15)

            self.num_epochs_completed += 1

            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")        

            if self.num_epochs_completed % save_frequency == 0:
                torch.save({

                    'epoch' : self.num_epochs_completed,
                    'model_state_dict': self.__model.state_dict()

                }, 'checkpoints/VAE_%d_%.2f.pt' % (self.num_epochs_completed, epoch_train_loss))
		
    """
    Handles testing the VAE. Requires a DataLoader which has been set to load only the
    testing dataset.
    """
    def test(self, test_loader):

        batch_size = 128

        self.__model.eval()
        test_loss = 0

        # Not computing gradients
        with torch.no_grad():
            for batch, _ in test_loader:

                batch = batch.to(self.__device)
                reconstructed_batch, mu, logvar = self.__model(batch)
                test_loss += self.__loss_function(reconstructed_batch, batch, mu, logvar).item()

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))

    """
    Runs input data through only the encoder portion. Returns vectors of mu and logvar.
    """
    def encode(self, x):

        self.__model.eval()

        with torch.no_grad():
            return self.__model.encode(x)

    """
    Runs data that has been sampled from the learned distribution of the encoder.
    """
    def decode(self, z):

        self.__mode.eval()

        with torch.no_grad():
            return self.__model.decode(z)

    """
    Loads model weights and the number of epochs the model has been trained for from disk.
    """
    def load_checkpoint(self, path):

        checkpoint = torch.load(path, map_location=self.__device)
        self.__model.load_state_dict(checkpoint['model_state_dict'])
        self.num_epochs_completed = checkpoint['epoch']

    def temp_test(self, test_loader):

        import matplotlib
        matplotlib.use('Agg')
        import pylab

        for batch in test_loader:
            # mu, logvar = self.encode(batch['image'])
            batch = batch['image'].to(self.__device, dtype=torch.float)
            img, mu, logvar = self.__model(batch)
           
            # img = img.
 
            pylab.imshow(img[0, 0, 64, : , :].cpu().detach().numpy(), cmap='gray')
            pylab.show()
            pylab.savefig('img.png')

            return
            

"""
Utility function to sample from a distribution that has been learned by the encoder.
"""
def sample_distribution(mu, logvar):

    logvar = torch.exp(0.5*logvar)
    eps = torch.randn_like(logvar)
    return mu + eps*eps
