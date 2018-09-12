import time
import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
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

levine_tsv = np.loadtxt("CYTOMETRY_data/levine.tsv", delimiter="\t", skiprows=1)
levine_data = levine_tsv[:,:levine_tsv.shape[1] - 1]
levine_labels = levine_tsv[:,levine_tsv.shape[1] - 1].astype(int)

print(levine_data.shape)

indices = np.random.permutation(levine_data.shape[0])
subset_a_indices = indices[:70000]
subset_b_indices = indices[70000:]

n_input_dimensions = levine_data.shape[1]
n_latent_dimensions = 2

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.1),
  gaussian_supplier)

fit_params = {
  "n_iters": 1500,
  "batch_size": 100,
  "deterministic": True,
  "fit_vae": True,
  "n_vae_iters": 10000,
  "vae_batch_size": 1000}

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
  perplexity=10)

ptsne = PTSNE(
  [n_input_dimensions],
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
  perplexity=10)

pca = PCA(n_components=2)
umap = UMAP(n_components=2)
tsne = TSNE(n_components=2, perplexity=400)

estimators = [tsne, pca]#[vptsne, ptsne, vae, pca, umap, tsne]

def fit_transform_fn(estimator):
  print("Running fit_transform with estimator", estimator.__class__.__name__)
  start = curr_millis()
  if isinstance(estimator, PTSNE):
    transformed = estimator.fit_transform(levine_data, **fit_params)
  if isinstance(estimator, VAE): # Already fitted
    transformed = estimator.transform(levine_data)
  else:
    transformed = estimator.fit_transform(levine_data)
  print(estimator.__class__.__name__, "fit_transform completed in", curr_millis() - start, "(ms)")
  return transformed

transformed_all = [fit_transform_fn(estimator) for estimator in estimators]

print(
  "Trustworthiness (vptsne, ptsne, vae, pca, umap, tsne)",
  [trustworthiness(levine_data[subset_b_indices], transformed[subset_b_indices], n_neighbors=12) for transformed in transformed_all])

print(
  "1-NN score for test set (vptsne, ptsne, vae, pca, umap, tsne)",
  [KNC(n_neighbors=1)
    .fit(transformed[subset_a_indices], levine_labels[subset_a_indices])
    .score(transformed[subset_b_indices], levine_labels[subset_b_indices])
    for transformed in transformed_all])

for i, transformed in enumerate(transformed_all):
  plt.clf()
  for label in np.unique(levine_labels):
    tmp = transformed[levine_labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], s=0.2, c=color_palette[label])
  plt.show()

