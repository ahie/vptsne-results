import time
import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
import fashion_mnist
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier as KNC

curr_millis = lambda: int(round(time.time() * 1000))

np.random.seed(0)
color_palette = np.random.rand(100, 3)

fashion_data, fashion_labels = fashion_mnist.train("FASHION_MNIST_data/")
fashion_data = fashion_data.reshape([-1, 784])

indices = np.random.permutation(fashion_data.shape[0])
subset_a_indices = indices[:55000]
subset_b_indices = indices[55000:]

n_input_dimensions = fashion_data.shape[1]
n_latent_dimensions = 3

encoder_hidden_3_shape = None
encoder_flattened_shape = None

def relu_bn(x, training):
  return tf.nn.relu(tf.layers.batch_normalization(x, training=training))

def encoder_network_builder(vae, x):
  hidden_1 = relu_bn(tf.layers.conv2d(x, 128, 2, [1, 1]), vae.training)
  hidden_2 = relu_bn(tf.layers.conv2d(hidden_1, 128 * 2, 2, [1, 1]), vae.training)
  hidden_3 = relu_bn(tf.layers.conv2d(hidden_2, 128 * 4, 2, [1, 1]), vae.training)
  flattened = tf.layers.Flatten()(hidden_3)

  global encoder_hidden_3_shape, encoder_flattened_shape
  encoder_hidden_3_shape = hidden_3.shape[1:]
  encoder_flattened_shape = flattened.shape[1]

  return {
    "mu": tf.layers.dense(flattened, n_latent_dimensions, activation=None),
    "log_sigma_sq": tf.layers.dense(flattened, n_latent_dimensions, activation=None)}

def decoder_network_builder(vae, z):
  hidden_0 = tf.layers.dense(z, encoder_flattened_shape, activation=tf.nn.relu)
  hidden_1 = tf.reshape(hidden_0, [-1, *encoder_hidden_3_shape])
  hidden_2 = relu_bn(tf.layers.conv2d_transpose(hidden_1, 128 * 4, 2, [1, 1]), vae.training)
  hidden_3 = relu_bn(tf.layers.conv2d_transpose(hidden_2, 128 * 2, 2, [1, 1]), vae.training)
  hidden_4 = relu_bn(tf.layers.conv2d_transpose(hidden_3, 128, 2, [1, 1]), vae.training)
  hidden_5 = tf.nn.sigmoid(tf.layers.conv2d(hidden_4, 1, 1, [1, 1]))
  return {
    "output": z,
    "probs": hidden_5}

if True:
  vae = VAE(
    [784],
    get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
    gaussian_prior_supplier,
    gaussian_supplier,
    get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
    bernoulli_supplier)
else:
  vae = VAE(
    [28, 28, 1],
    encoder_network_builder,
    gaussian_prior_supplier,
    gaussian_supplier,
    decoder_network_builder,
    bernoulli_supplier,
    learning_rate=0.00001)

fit_params = {
  "hook_fn": print,
  "n_iters": 1500,
  "batch_size": 500,
  "deterministic": True,
  "fit_vae": False,
  "n_vae_iters": 5000,
  "vae_batch_size": 100}

def disp(n):
  for i in range(n):
    plt.subplot(211)
    plt.imshow(fashion_data[i].reshape((28,28)))
    plt.subplot(212)
    plt.imshow(vae.reconstruct([fashion_data[i]])[0].reshape((28,28)))
    plt.show()

#vae.fit(fashion_data, n_iters=10000, batch_size=1000, hook_fn=print)
#vae.save_weights("models/fashion_vae_non_cnn.ckpt")

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
  perplexity=30)

ptsne = PTSNE(
  [784],
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
  perplexity=30)

pca = PCA(n_components=2)
umap = UMAP(n_components=2)
tsne = TSNE(n_components=2, perplexity=30)

estimators = [ptsne]#, ptsne, vae]#, pca, umap, tsne]

def fit_transform_fn(estimator):
  print("Running fit_transform with estimator", estimator.__class__.__name__)
  start = curr_millis()
  if isinstance(estimator, PTSNE):
    #estimator.load_weights("models/fashion_vptsne.ckpt")
    estimator.fit(fashion_data, **fit_params)
    estimator.save_weights("models/fashion_ptsne.ckpt")
    transformed = estimator.transform(fashion_data)
  elif isinstance(estimator, VAE): # Already trained
    transformed = estimator.transform(fashion_data)
  else:
    transformed = estimator.fit_transform(fashion_data)
  print(estimator.__class__.__name__, "fit_transform completed in", curr_millis() - start, "(ms)")
  return transformed

transformed_all = [fit_transform_fn(estimator) for estimator in estimators]

print(
  "Trustworthiness (vptsne, ptsne, vae, pca, umap, tsne)",
  [trustworthiness(fashion_data[subset_b_indices], transformed[subset_b_indices], n_neighbors=12) for transformed in transformed_all])

print(
  "1-NN score for test set (vptsne, ptsne, vae, pca, umap, tsne)",
  [KNC(n_neighbors=1)
    .fit(transformed[subset_a_indices], fashion_labels[subset_a_indices])
    .score(transformed[subset_b_indices], fashion_labels[subset_b_indices])
    for transformed in transformed_all])

for i, transformed in enumerate(transformed_all):
  plt.clf()
  for label in np.unique(fashion_labels):
    tmp = transformed[fashion_labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], s=0.05, c=color_palette[label])
  plt.show()

