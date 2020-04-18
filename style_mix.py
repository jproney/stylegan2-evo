import torch
from torchvision import utils
from stylegan2.model import Generator


def generate(args, g_ema, device, mean_latent):
