import torch
from stylegan2.model import Generator
from stylegan2 import ppl
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torchvision
import math

LATENT_DIM = 512
IMG_DIM = 1024


class Evolver:
    def __init__(self, npop, target, cpkt = "stylegan2/stylegan2-ffhq-config-f.pt", lamb=5.0, sigma=.05, device="cpu", trunc=.6):
        with torch.no_grad():
            self.device = device
            self.npop = npop
            self.target = target
            self.lamb = lamb
            self.sigma = sigma
            self.generator = Generator(IMG_DIM, LATENT_DIM, 8, channel_multiplier=2).to(self.device)
            self.generator.eval()
            checkpoint = torch.load(cpkt)
            self.generator.load_state_dict(checkpoint['g_ema'])
            sample = torch.randn(npop, LATENT_DIM, device=device)
            trunc_target = self.generator.mean_latent(4096)
            # matrix of genomes
            self.genomes = trunc_target + trunc*(self.generator.get_latent(sample) - trunc_target)
            self.fitness = None
            self.faces = None
            self.generate()

    def generate(self):
        with torch.no_grad():
            self.faces, _ = self.generator([self.genomes], input_is_latent=True, truncation=1)

    def calc_error(self):
        with torch.no_grad():
            mse = torch.nn.MSELoss()
            self.fitness = torch.exp(torch.tensor([-mse(self.faces[i, :, :, :], self.target) for i in range(self.npop)]))

    def update(self):
        with torch.no_grad():
            self.calc_error()
            tot_fit = torch.sum(self.fitness)
            child_genomes = torch.zeros([self.npop, LATENT_DIM], device=self.device)
            par1 = torch.multinomial(self.fitness/tot_fit, self.npop)
            par2 = torch.multinomial(self.fitness/tot_fit, self.npop)
            noise1 = torch.randn(self.npop, LATENT_DIM, device=self.device) * self.sigma
            noise2 = torch.randn(self.npop, LATENT_DIM, device=self.device) * self.sigma
            for i in range(self.npop):
                # mutate the parent genomes
                w1 = self.genomes[par1[i], :] + noise1[par1[i], :]
                w2 = self.genomes[par2[i], :] + noise2[par2[i], :]

                # cross over
                xpoints = [0] + sorted(random.sample(range(1, LATENT_DIM), 2)) + [LATENT_DIM]
                currpar = True
                for j in range(len(xpoints) - 1):
                    if currpar:
                        child_genomes[i, xpoints[j]:xpoints[j+1]] = w1[xpoints[j]:xpoints[j+1]]
                        currpar = False
                    else:
                        child_genomes[i, xpoints[j]:xpoints[j+1]] = w2[xpoints[j]:xpoints[j+1]]
                        currpar = True
            self.genomes = child_genomes
            self.generate()


    def display(self, ndisp):
        with torch.no_grad():
            dim = math.ceil(math.sqrt(ndisp))
            fig, axs = plt.subplots(dim, dim)
            disp_ind = random.sample(range(self.npop), ndisp)
            for i in range(ndisp):
                row = i // dim
                im = self.faces[disp_ind[i], :, :, :].cpu().numpy().squeeze()
                im = np.moveaxis(im, [0, 1, 2], [2, 0, 1])
                axs[row, i - row*dim].imshow(im*.5 + .5)
            plt.show()


if __name__ == "__main__":
    with torch.no_grad():
        n_iter = 100
        img = Image.open("target2.png")
        totens = torchvision.transforms.ToTensor()
        tgt = totens(img)
        evo = Evolver(6, tgt)
        # for i in range(n_iter):
        #     evo.update()
        #     evo.display()






