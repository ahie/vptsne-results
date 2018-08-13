import time
import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
import hdata
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier as KNC

curr_millis = lambda: int(round(time.time() * 1000))

np.random.seed(0)
color_palette = np.random.rand(100, 3)

data, labels = hdata.load_all("CYTOMETRY_data")
print(data.shape, labels.shape)

indices = np.random.permutation(data.shape[0])
number_of_points = int(data.shape[0] * 0.01)
subset_indices = indices[:number_of_points]

n_input_dimensions = data.shape[1]
n_latent_dimensions = 5

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.1),
  gaussian_supplier)

fit_params = {
  "hook_fn": print,
  "n_iters": 1500,
  "batch_size": 5000,
  "deterministic": True,
  "fit_vae": True,
  "n_vae_iters": 20000,
  "vae_batch_size": 10000}

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=True),
  perplexity=10)

pca = PCA(n_components=2)
umap = UMAP(n_components=2)
tsne = TSNE(n_components=2, perplexity=10)

estimators = [vptsne, umap]

def fit_fn(estimator):
  print("Running fit with estimator", estimator.__class__.__name__)
  start = curr_millis()
  if isinstance(estimator, PTSNE):
    transformed = estimator.fit(data, **fit_params)
  else:
    transformed = estimator.fit(data)
  print(estimator.__class__.__name__, "fit completed in", curr_millis() - start, "(ms)")
  return transformed

[fit_fn(estimator) for estimator in estimators]

