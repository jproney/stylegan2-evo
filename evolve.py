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
CUDA_BATCH_SIZE = 6  # Deep Learning on a laptop reacs only
N_FLOW = 16

class Evolver:
    def __init__(self, npop, target, cpkt = "stylegan2/stylegan2-ffhq-config-f.pt", lamb=5.0, sigma=0.4, device="cuda", trunc=.6, mask=None):
        with torch.no_grad():
            self.mask = mask
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
            self.trunc_target = self.generator.mean_latent(4096)
            # matrix of genomes
            self.trunc = trunc
            self.genomes = self.trunc_target + self.trunc*(self.generator.get_latent(sample) - self.trunc_target)
            self.fitness = None
            self.ranks = torch.zeros([npop], device="cpu")
            self.faces = torch.zeros([npop, 3, IMG_DIM, IMG_DIM], device="cpu")
            self.generate()

    def generate(self):
        with torch.no_grad():
            i = 0
            while i < self.npop:
                torch.cuda.empty_cache()
                nxt = min(i + CUDA_BATCH_SIZE, self.npop)
                self.faces[i:nxt, :, :, :] = self.generator([self.genomes[i:nxt, :]], input_is_latent=True, truncation=1)[0].cpu()
                print("we got one!")
                i += CUDA_BATCH_SIZE

    def calc_error(self):
        with torch.no_grad():
            mse = torch.nn.MSELoss()
            if self.mask is None:
                self.fitness = [(i, mse(self.faces[i, :, :, :], self.target)) for i in range(self.npop)]
            else:
                x1, x2, y1, y2 = self.mask
                self.fitness = [(i, mse(self.faces[i, :, x1:x2, y1:y2], self.target[:, x1:x2, y1:y2])) for i in range(self.npop)]
            self.fitness.sort(key=lambda elem: elem[1])
            for i in range(self.npop):
                x = self.fitness[i]
                self.ranks[x[0]] = self.npop - i

    def update(self):
        with torch.no_grad():
            self.calc_error()
            tot_fit = torch.sum(self.ranks)
            child_genomes = torch.zeros([self.npop, LATENT_DIM], device=self.device)
            par1 = torch.multinomial(self.ranks/tot_fit, self.npop - 1, replacement=True)
            par2 = torch.multinomial(self.ranks/tot_fit, self.npop - 1, replacement=True)
            noise1 = torch.randn(self.npop - 1, LATENT_DIM, device=self.device)
            noise2 = torch.randn(self.npop - 1, LATENT_DIM, device=self.device)
            noise1 = self.generator.get_latent(noise1)
            noise2 = self.generator.get_latent(noise2)
            child_genomes[0, :] = self.genomes[self.fitness[0][0], :]  # Do elitism
            for i in range(1, self.npop):
                # mutate the parent genomes by offsetting with a random face
                w1 = self.genomes[par1[i-1], :] + .05*(noise1[i-1, :] - self.trunc_target)
                w2 = self.genomes[par2[i-1], :] + .05*(noise2[i-1, :] - self.trunc_target)

                # cross over
                xpoint = random.randint(1, LATENT_DIM)
                child_genomes[i, 0:xpoint] = w1[0, 0:xpoint]
                child_genomes[i, xpoint:] = w2[0, xpoint:]
            self.genomes = child_genomes
            self.generate()


    def display(self, ndisp):
        with torch.no_grad():
            dim = math.ceil(math.sqrt(ndisp))
            fig, axs = plt.subplots(dim, dim)
            for i in range(ndisp):
                row = i // dim
                ind = self.fitness[i][0]
                im = self.faces[ind, :, :, :].cpu().numpy().squeeze()
                im = np.moveaxis(im, [0, 1, 2], [2, 0, 1])
                axs[row, i - row*dim].imshow(im*.5 + .5)
            plt.show()


if __name__ == "__main__":
    with torch.no_grad():
        n_iter = 100
        img = Image.open("target2.png")
        totens = torchvision.transforms.ToTensor()
        tgt = totens(img)
        evo = Evolver(128, tgt, mask=(350, 820, 300, 720))
        # for i in range(n_iter):
        #     evo.update()
        #     evo.display()






