import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.neighbors import KNeighborsClassifier as KNC
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(0)
color_palette = np.random.rand(100, 3)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
n_input_dimensions = mnist.train._images.shape[1]
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
#  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.1),
#  gaussian_supplier,
  get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
  bernoulli_supplier,
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
#  "hook_fn": print,
  "n_iters": 1500,
  "batch_size": 300,
  "fit_vae": True,
  "n_vae_epochs": 200,
  "vae_batch_size": 1000}
vptsne.fit(mnist.train._images, **fit_params)
ptsne.fit(mnist.train._images, **fit_params)
vptsne.save_weights("models/mnist_vptsne.ckpt", "models/mnist_vae.ckpt")
ptsne.save_weights("models/mnist_ptsne.ckpt")

#vptsne.load_weights("models/mnist_vptsne.ckpt", "models/mnist_vae.ckpt")
#ptsne.load_weights("models/mnist_ptsne.ckpt")

pca = PCA(n_components=2).fit(mnist.train._images)

estimators = [vptsne, ptsne, vae, pca]
transformed_train = [estimator.transform(mnist.train._images) for estimator in estimators]
transformed_test = [estimator.transform(mnist.test._images) for estimator in estimators]

print(
  "Trustworthiness for test set (vptsne, ptsne, vae, pca):",
  [trustworthiness(mnist.test._images, transformed, n_neighbors=12) for transformed in transformed_test])

print(
  "1-NN score for test set (vptsne, ptsne, vae, pca)",
  [KNC(n_neighbors=1)
    .fit(train, mnist.train._labels)
    .score(test, mnist.test._labels)
    for train, test in zip(transformed_train, transformed_test)])

