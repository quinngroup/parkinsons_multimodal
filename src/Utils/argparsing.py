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
    parser.add_argument("-d", "--download", action='store_true',
    help = "Whether to download data from cloud bucket. If True, then must provide -k arg. [DEFAULT: False]")

    parser.add_argument("-k", "--key_path", default = None, type = str,
    help = "Path to .json file containing credential to access cloud bucket. Only used if -d is specified. [DEFAULT: None]")

    parser.add_argument("--bucket", default = "ppmi", type = str,
    help = "Name of cloud bucket containing data. [DEFAULT: ppmi]")

    parser.add_argument("-b", "--batch_size", default = 8, type = int,
    help = "Batch sizes of data that will be fed to the model. [DEFAULT: 64]")

    parser.add_argument("--train", action='store_true',
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

    # If user wants to download, but does not provide the path to cloud bucket key file
    if args['download'] and args['key_path'] is None:
        parser.error("--download flag requires --key_path.")

    return args
