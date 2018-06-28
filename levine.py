import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.neighbors import KNeighborsClassifier as KNC

np.random.seed(0)
color_palette = np.random.rand(100, 3)

levine_tsv = np.loadtxt("CYTOMETRY_data/levine.tsv", delimiter="\t", skiprows=1)
levine_data = levine_tsv[:,:levine_tsv.shape[1] - 1]
levine_labels = levine_tsv[:,levine_tsv.shape[1] - 1].astype(int)

n_input_dimensions = levine_train_images.shape[1]
n_latent_dimensions = 2

vae_layer_definitions = [
  (256, tf.nn.relu),
  (128, tf.nn.relu),
  (32, tf.nn.relu)]
vae_encoder_layers = LayerDefinition.from_array(vae_layer_definitions)
vae_decoder_layers = LayerDefinition.from_array(reversed(vae_layer_definitions))

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.025),
  gaussian_supplier,
  beta=1.0)

vptsne_layers = LayerDefinition.from_array([
  (250, tf.nn.relu),
  (2500, tf.nn.relu),
  (2, None)])

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
  perplexity=30.)

ptsne = PTSNE(
  [n_input_dimensions],
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
  perplexity=30.)

fit_params = {
  "hook_fn": print,
  "n_iters": 1500,
  "batch_size": 300,
  "deterministic": True,
  "fit_vae": True,
  "n_vae_epochs": 200,
  "vae_batch_size": 1000}
vptsne.fit(levine_train_images, **fit_params)
ptsne.fit(levine_train_images, **fit_params)
vptsne.save_weights("models/levine_vptsne.ckpt", "models/levine_vae.ckpt")
ptsne.save_weights("models/levine_ptsne.ckpt")

#vptsne.load_weights("models/levine_vptsne.ckpt", "models/levine_vae.ckpt")
#ptsne.load_weights("models/levine_ptsne.ckpt")

pca = PCA(n_components=2).fit(levine_train_images)

estimators = [vptsne, ptsne, vae, pca]
transformed_train = [estimator.transform(levine_train_images) for estimator in estimators]
transformed_test = [estimator.transform(levine_test_images) for estimator in estimators]

print(
  "Trustworthiness for test set (vptsne, ptsne, vae, pca):",
  [trustworthiness(levine_test_images, transformed, n_neighbors=12) for transformed in transformed_test])

print(
  "1-NN score for test set (vptsne, ptsne, vae, pca)",
  [KNC(n_neighbors=1)
    .fit(train, levine_train_labels)
    .score(test, levine_test_labels)
    for train, test in zip(transformed_train, transformed_test)])

