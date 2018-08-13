import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.neighbors import KNeighborsClassifier as KNC
from tensorflow.examples.tutorials.mnist import input_data

n_input_dimensions = mnist_train_images.shape[1]

def run_training(n_latent_dimensions, perplexity, batch_size, run_id):

  info = "%d_%d_%d_%d" % (n_latent_dimensions, perplexity, batch_size, run_id)

  vae = VAE(
    [n_input_dimensions],
    get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
    gaussian_prior_supplier,
    gaussian_supplier,
    get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
    bernoulli_supplier)
  
  vptsne = VPTSNE(
    vae,
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity,
    learning_rate=0.001)
  
  ptsne = PTSNE(
    [n_input_dimensions],
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity,
    learning_rate=0.001)

  def get_logger(loss_file, trustworthiness_file, knn_file):
    def log_fn(args):
      if isinstance(args[0], VAE):
        return
      loss_file.write(str(args[2]) + "\n")
      if args[1] % 50 == 0:
        transformed_train = args[0].transform(mnist_train_images)
        transformed_test = args[0].transform(mnist_test_images)
        trustworthiness_file.write(str(
          trustworthiness(mnist_test_images, transformed_test, n_neighbors=12)) + "\n")
        knn_file.write(str(
          KNC(n_neighbors=1)
          .fit(transformed_train, mnist_train_labels)
          .score(transformed_test, mnist_test_labels)) + "\n")
    return log_fn

  fit_params = {
    "n_iters": 2000,
    "batch_size": batch_size,
    "fit_vae": True,
    "n_vae_iters": 10000,
    "vae_batch_size": 1000}

  vptsne_log_files = [open("parameter_tuning_output/%s_vptsne_%s.log" % (to_log, info), "w") for to_log in ["loss", "trustworthiness", "knn"]]
  ptsne_log_files = [open("parameter_tuning_output/%s_ptsne_%s.log" % (to_log, info), "w") for to_log in ["loss", "trustworthiness", "knn"]]

  vptsne.fit(mnist_train_images, hook_fn=get_logger(*vptsne_log_files), **fit_params)
  ptsne.fit(mnist_train_images, hook_fn=get_logger(*ptsne_log_files), **fit_params)

  vptsne.save_weights("models/mnist_vptsne_%s.ckpt" % info, "models/mnist_vae_%s.ckpt" % info)
  ptsne.save_weights("models/mnist_ptsne_%s.ckpt" % info)

  for f in vptsne_log_files:
    f.close()
  for f in ptsne_log_files:
    f.close()

if __name__ == "__main__":
  for perplexity in [30]:
    for n_latent_dimensions in [3, 5]:
      for batch_size in [200, 400]:
        for run_id in range(20):
          run_training(n_latent_dimensions, perplexity, batch_size, run_id)
      for batch_size in [200, 400, 800]:
        for run_id in range(20, 40):
          run_training(n_latent_dimensions, perplexity, batch_size, run_id)

