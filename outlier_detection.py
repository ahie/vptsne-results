import numpy as np
import matplotlib.pyplot as plt
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from sklearn.preprocessing import MinMaxScaler

n_input_dimensions = mnist_train_images.shape[1]
n_latent_dimensions = 3

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
  bernoulli_supplier)

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False))

fit_params = {
  "hook_fn": print,
  "n_iters": 1500,
  "batch_size": 400,
  "fit_vae": True,
  "n_vae_iters": 10000,
  "vae_batch_size": 1000}

vptsne.fit(mnist_train_images, **fit_params)
transformed = vptsne.transform(mnist_test_images[:3000])
scores = MinMaxScaler().fit_transform(vae.score(mnist_test_images[:3000]).reshape(-1, 1))

plt.scatter(transformed[:, 0], transformed[:, 1], c=np.apply_along_axis(lambda x: [x[0], 0, 0, 0.5], 1, scores))
plt.show()

