import argparse

"""
Returns args parsed from the command line. Handles optional and required args
as well as if/else relationships between args.
"""
def get_args():

    # Setting up argparser
    parser = argparse.ArgumentParser(description = "Parkinson's disease research",
    epilog = "Quinn Research Group",
    add_help = "How to use")

    # Optional args
    parser.add_argument("-b", "--batch_size", default = 1, type = int,
    help = "Batch sizes of data that will be fed to the model. [DEFAULT: 8]")

    parser.add_argument("--train", action='store_true', default = True,
    help = "Whether to train the model. [DEFAULT: False]")

    parser.add_argument("-l", "--latent_size", default = 256, type = int,
        help = "Size of latent vectors learned by the VAE. [DEFAULT: 256]")

    parser.add_argument("--save_frequency", default = 5, type = int,
        help = "Number of epochs after which the VAE will be save. [DEFAULT: 5]")

    parser.add_argument("-e", "--epochs", default = 5, type = int,
        help = "Number of epochs for which the VAE will be trained for. [DEFAULT: 5]")

    parser.add_argument("--load_checkpoint", default=None,
        help = "Path to VAE model that will be loaded. If no path is specified, new VAE will be created. [DEFAUJLT: None]")

    args = vars(parser.parse_args())

    return args
