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

def run_training(n_latent_dimensions, perplexity, batch_size, run_id):

  info = "%d_%d_%d_%d" % (n_latent_dimensions, perplexity, batch_size, run_id)

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
    get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
    bernoulli_supplier)

  vptsne_layers = LayerDefinition.from_array([
    (200, tf.nn.relu),
    (200, tf.nn.relu),
    (2000, tf.nn.relu),
    (2, None)])
  
  vptsne = VPTSNE(
    vae,
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity)
  
  ptsne = PTSNE(
    [n_input_dimensions],
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity)

  def get_logger(loss_file, trustworthiness_file, knn_file):
    def log_fn(args):
      if isinstance(args[0], VAE):
        return
      loss_file.write(str(args[2]) + "\n")
      if args[1] % 10 == 0:
        transformed_train = args[0].transform(mnist.train._images)
        transformed_test = args[0].transform(mnist.test._images)
        trustworthiness_file.write(str(
          trustworthiness(mnist.test._images, transformed_test, n_neighbors=12)) + "\n")
        knn_file.write(str(
          KNC(n_neighbors=1)
          .fit(transformed_train, mnist.train._labels)
          .score(transformed_test, mnist.test._labels)) + "\n")
    return log_fn

  fit_params = {
    "n_iters": 1500,
    "batch_size": batch_size,
    "fit_vae": True,
    "n_vae_epochs": 200,
    "vae_batch_size": 1000}

  vptsne_log_files = [open("output/%s_vptsne_%s.log" % (to_log, info), "w") for to_log in ["loss", "trustworthiness", "knn"]]
  ptsne_log_files = [open("output/%s_ptsne_%s.log" % (to_log, info), "w") for to_log in ["loss", "trustworthiness", "knn"]]

  vptsne.fit(mnist.train._images, hook_fn=get_logger(*vptsne_log_files), **fit_params)
  ptsne.fit(mnist.train._images, hook_fn=get_logger(*ptsne_log_files), **fit_params)

  vptsne.save_weights("models/mnist_vptsne_%s.ckpt" % info, "models/mnist_vae_%s.ckpt" % info)
  ptsne.save_weights("models/mnist_ptsne_%s.ckpt" % info)

  for f in vptsne_log_files:
    f.close()
  for f in ptsne_log_files:
    f.close()

if __name__ == "__main__":
  for perplexity in [30]:
    for n_latent_dimensions in [4, 3, 2]:
      for batch_size in [200, 400, 800]:
        for run_id in range(10):
          run_training(n_latent_dimensions, perplexity, batch_size, run_id)

