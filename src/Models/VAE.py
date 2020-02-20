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
            Downsample()

        )

        self.fc_mu = FC(1024, latent_size)
        self.fc_logvar = FC(1024, latent_size)

        self.decoder_linear = FC(latent_size, 1024)

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
    Encodes image, takes a sample from learned distribution, then decodes it.
    """
    def forward(self, x):

        x = self.encoder_convolutions(x)
        x = x.view(x.size()[0], -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = reparameterize(mu, logvar)

        z = self.decoder_linear(z)
        z = z.view(z.size()[0], -1, 1, 1, 1)
        z = self.decoder_convolutions(z)

        return z, mu, logvar
        

"""
Controller for the VAE itself. Handles training, testing, encoding, and decoding.
"""
class VAE():
	
    """
    Handles initializing the VAE and determines correct device to be run on.
    """
    def __init__(self, latent_size, lr=1e-2):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = nn.DataParallel(_VAE_NN(latent_size))

        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.num_epochs_completed = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.latent_size = latent_size

        print("[INFO] Device detected: %s" % self.device)

    """
    Calculates the loss function of the VAE encoding and decoding input. Loss is 
    given by calculating the mean squared error loss between the reconstructed input
    and actual input, then adding it to the KL Divergence which measures differences in the 
    distribution of data.
    """
    def __loss_function(self, recon_x, x, mu, logvar):

        # TODO Rethink this
        MSE = F.mse_loss(x, recon_x)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

    """
    Handles training the VAE. Requires a DataLoader which has been set to load only the
    training dataset.
    """
    def train(self, train_loader, epochs, save_frequency):

        self.model.train()
        print("[INFO] Beginning VAE training")

        # Training for selected number of epochs
        for epoch in range(1 + self.num_epochs_completed, epochs + 1 + self.num_epochs_completed):

            train_loss = 0

            # Looping through data batches from the loader
            for batch_idx, batch_data in enumerate(train_loader):

                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                # Passing batch through VAE
                batch = batch_data['image'].to(self.device, dtype=torch.float)
                reconstructed_batch, mu, logvar = self.model(batch.float())

                # Calculating and backpropogating error through the model
                loss = self.__loss_function(reconstructed_batch, batch, mu, logvar).to('cpu')
                loss.backward()
                train_loss += float(loss)

                # Changing model weights to minimize loss
                self.optimizer.step()

                # Logging
                self.log_batch(epoch, loss.item(), batch_idx, len(batch), len(train_loader.dataset), len(train_loader))

            epoch_train_loss = train_loss / len(train_loader.dataset)

            # Logging
            print('\r====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_train_loss) + " " * 25)

            self.num_epochs_completed += 1

            # Checkpoint reached
            if self.num_epochs_completed % save_frequency == 0:
                self.handle_checkpoint(epoch_train_loss)
		
    """
    Handles testing the VAE. Requires a DataLoader which has been set to load only the
    testing dataset.
    """
    def test(self, test_loader):

        batch_size = 128

        self.model.eval()
        test_loss = 0

        # Not computing gradients
        with torch.no_grad():
            for batch, _ in test_loader:

                batch = batch.to(self.device)
                reconstructed_batch, mu, logvar = self.model(batch)
                test_loss += self.__loss_function(reconstructed_batch, batch, mu, logvar).item()

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))

    """
    Loads model weights and the number of epochs the model has been trained for from disk.
    """
    def load_checkpoint(self, path):

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_epochs_completed = checkpoint['epoch']
        # self.latent_size = checkpoint['latent_size']

        print('[INFO] Loaded model')
        print('*' * 15)
        print('Epochs: %d\nLatent size: %d' % 
                (self.num_epochs_completed, self.latent_size))
        print('*' * 15)

    def visualization(self, test_loader):

        import matplotlib
        matplotlib.use('Agg')
        import pylab

        for batch in test_loader:

            batch = batch['image'].to(self.device, dtype=torch.float)
            img, mu, logvar = self.model(batch)
           
            pylab.imshow(img[0, 0, 64, : , :].cpu().detach().numpy(), cmap='gray')
            pylab.show()
            pylab.savefig('img.png')

            return
            
    """
    Passes input data through the VAE. Returns the reconstructed input, a latent vector, and the mu and
    logvar that are used to get the latent vectors.
    """
    def forward(self, x):

        self.model.eval()

        with torch.no_grad():
            y, mu, logvar = self.model.forward(x)
            z = reparameterize(mu, logvar)
            return y, z, mu, logvar

    """
    Logging model training performance for each batch within the dataset.
    """
    def log_batch(self, epoch, loss, batch_idx, batch_len, dataset_len, loader_len):
        sys.stdout.write('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

            epoch, batch_idx * batch_len, dataset_len,
            100. * batch_idx / loader_len,
            loss / batch_len)
            
        )

        sys.stdout.flush()

    """
    Writes necessary portions of the model to disk for checkpointing. Will create checkpoints/ folder
    if it doesn't exist and place all checkpoints there.

    Name will be VAE_[EPOCH]_[LOSS].pt.
    """
    def handle_checkpoint(self, epoch_train_loss):

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        torch.save({

            'epoch' : self.num_epochs_completed,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_size': self.latent_size

        }, 'checkpoints/VAE_%d_%.2f.pt' % (self.num_epochs_completed, epoch_train_loss))


"""
Utility function to sample from a distribution that has been learned by the encoder.
"""
def reparameterize(mu, logvar):

    logvar = torch.exp(0.5*logvar)
    eps = torch.randn_like(logvar)
    return mu + eps*eps
