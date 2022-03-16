import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchaudio

from models import CNNModel, GramMatrix, StyleLoss

def main(args):
    # Load audios

    # Check sampling rate?

    # Reshape

    # Move to GPU if available

    # define cnn model + move to GPU

    # get style and model loss

    # optimizer

    # style transfer
    pass

if __name__ == "main":
    parser = ArgumentParser()
    parser.add_argument('--script', type=str)
    parser.add_argument('--content_audio', type=str)
    parser.add_argument('--style_audio', type=str)
    args = parser.parse_args()

    main(args)