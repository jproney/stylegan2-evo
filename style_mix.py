import torch
from stylegan2.model import Generator
from stylegan2 import ppl
import matplotlib.pyplot as plt
import numpy as np

# Test out style mixing, interpolation

# Load the model
device = "cuda"
g_ema = Generator(1024, 512, 8, channel_multiplier=2).to(device)
checkpoint = torch.load("stylegan2/stylegan2-ffhq-config-f.pt")
g_ema.load_state_dict(checkpoint['g_ema'])


with torch.no_grad():
    # Try interpolating
    latent1 = torch.randn(1, 512, device=device)
    latent2 = torch.randn(1, 512, device=device)
    mix = ppl.slerp(latent1, latent2, .5)

    trunc_target = g_ema.mean_latent(4096)

    # Do some generating
    sample1, _ = g_ema([latent1], truncation=.6, truncation_latent=trunc_target)
    sample2, _ = g_ema([latent2], truncation=.6, truncation_latent=trunc_target)
    sample_mixed, _ = g_ema([mix], truncation=.6, truncation_latent=trunc_target)

    sample1 = sample1.cpu().numpy().squeeze()
    sample1 = np.moveaxis(sample1, [0, 1, 2], [2, 0, 1])
    sample2 = sample2.cpu().numpy().squeeze()
    sample2 = np.moveaxis(sample2, [0, 1, 2], [2, 0, 1])
    sample_mixed = sample_mixed.cpu().numpy().squeeze()
    sample_mixed = np.moveaxis(sample_mixed, [0, 1, 2], [2, 0, 1])

    fig, axis = plt.subplots(1, 3)
    axis[0].imshow(sample1*.5 + .5)
    axis[1].imshow(sample_mixed*.5 + .5)
    axis[2].imshow(sample2*.5 + .5)
    plt.show()

    # Now try doing crossover in the post-mlp embedding space
    w1 = trunc_target + .6*(g_ema.get_latent(latent1) - trunc_target) # truncate first, then do crossover
    w2 = trunc_target + .6*(g_ema.get_latent(latent2) - trunc_target)
    cross1 = torch.cat((w1[:, :128], w2[:, 128:]), 1)
    cross2 = torch.cat((w1[:, :256], w2[:, 256:]), 1)
    cross3 = torch.cat((w2[:, :256], w1[:, 256:]), 1)
    cross4 = torch.cat((w2[:, :128], w1[:, 128:]), 1)

    res1, _ = g_ema([w2], input_is_latent=True, truncation=1)
    res2, _ = g_ema([cross1], input_is_latent=True, truncation=1)
    res3, _ = g_ema([cross2], input_is_latent=True, truncation=1)
    res4, _ = g_ema([cross3], input_is_latent=True, truncation=1)
    res5, _ = g_ema([cross4], input_is_latent=True, truncation=1)
    res6, _ = g_ema([w1], input_is_latent=True, truncation=1)

    res1 = res1.cpu().numpy().squeeze()
    res1 = np.moveaxis(res1, [0, 1, 2], [2, 0, 1])
    res2 = res2.cpu().numpy().squeeze()
    res2 = np.moveaxis(res2, [0, 1, 2], [2, 0, 1])
    res3 = res3.cpu().numpy().squeeze()
    res3 = np.moveaxis(res3, [0, 1, 2], [2, 0, 1])
    res4 = res4.cpu().numpy().squeeze()
    res4 = np.moveaxis(res4, [0, 1, 2], [2, 0, 1])
    res5 = res5.cpu().numpy().squeeze()
    res5 = np.moveaxis(res5, [0, 1, 2], [2, 0, 1])
    res6 = res6.cpu().numpy().squeeze()
    res6 = np.moveaxis(res6, [0, 1, 2], [2, 0, 1])

    fig, axis = plt.subplots(1, 6)
    axis[0].imshow(res1*.5 + .5)
    axis[1].imshow(res2*.5 + .5)
    axis[2].imshow(res3*.5 + .5)
    axis[3].imshow(res4*.5 + .5)
    axis[4].imshow(res5*.5 + .5)
    axis[5].imshow(res6*.5 + .5)
    plt.show()

    # What if we just corrupt with latent representation with some random features?
    z = torch.randn(1, 512, device=device)
    ns = g_ema.get_latent(z)
    res_noisy, _ = g_ema([ns + .85*(w1 - ns)], input_is_latent=True, truncation=1)
    res_noisy = res_noisy.cpu().numpy().squeeze()
    res_noisy = np.moveaxis(res_noisy, [0, 1, 2], [2, 0, 1])
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(res6*.5 + .5)
    axis[1].imshow(res_noisy*.5 + .5)
    plt.show()





