import torch
from stylegan2.model import Generator
from stylegan2 import ppl
import matplotlib.pyplot as plt
import numpy as np
import random

LATENT_DIM = 512
IMG_DIM = 1024


class Evolver:
    def __init__(self, npop, target, cpkt = "stylegan2/stylegan2-ffhq-config-f.pt", lamb=3, sigma=.05, device="cuda", trunc=.6):
        with torch.no_grad():
            self.device = device
            self.npop = npop
            self.target = target
            self.lamb = lamb
            self.sigma = sigma
            self.generator = Generator(IMG_DIM, LATENT_DIM, 8, channel_multiplier=2).to(self.device)
            checkpoint = torch.load(cpkt)
            self.generator.load_state_dict(checkpoint['g_ema'])
            sample = torch.randn(npop, LATENT_DIM, device=device)
            trunc_target = self.generator.mean_latent(4096)
            # matrix of genomes
            self.genomes, _ = trunc_target + trunc*(self.generator.get_latent(sample) - trunc_target)
            self.losses = None
            self.faces = None
            self.generate()

    def generate(self):
        self.faces = self.generator([self.genomes], input_is_latent=True, truncation=1)

    def calc_error(self):
        mse = torch.nn.MSELoss()
        self.losses = torch.tensor([mse(self.faces[i, :, :], self.target) for i in range(self.npop)])

    def update(self):
        with torch.no_grad():
            self.generate()
            self.calc_error()
            tot_err = np.sum(self.losses)
            child_genomes = torch.zeros([self.npop, LATENT_DIM])
            par1 = torch.multinomial(self.losses/tot_err, self.npop)
            par2 = torch.multinomial(self.losses/tot_err, self.npop)
            noise1 = torch.randn(self.npop, LATENT_DIM) * self.sigma
            noise2 = torch.randn(self.npop, LATENT_DIM) * self.sigma
            for i in range(self.npop):
                # mutate the parent genomes
                w1 = self.genomes[par1[i], :] + noise1[par1[i], :]
                w2 = self.genomes[par2[i], :] + noise2[par2[i], :]

                # cross over
                k = torch.poisson(torch.tensor([self.lamb], device=self.device))
                xpoints = [0] + random.sample(range(1, LATENT_DIM-1), k) + [LATENT_DIM]
                par1 = True
                for j in xpoints[:-1]:
                    if par1:
                        child_genomes[i, xpoints[j]:xpoints[j+1]] = w1[xpoints[j]:xpoints[j+1]]
                        par1 = False
                    else:
                        child_genomes[i, xpoints[j]:xpoints[j+1]] = w2[xpoints[j]:xpoints[j+1]]
                        par1 = True

    def display(self):
        with torch.no_grad():
            pass









