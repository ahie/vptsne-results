import mnist
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

mnist_train_images, mnist_train_labels = mnist.train_images().reshape(60000, 784) / 255, mnist.train_labels()
mnist_test_images, mnist_test_labels = mnist.test_images().reshape(10000, 784) / 255, mnist.test_labels()
n_input_dimensions = mnist_train_images.shape[1]
n_latent_dimensions = 3

def run_training(vae, n_latent_dimensions, perplexity, batch_size, run_id):

  info = "%d_%d_%d_%d" % (n_latent_dimensions, perplexity, batch_size, run_id)
  
  vptsne = VPTSNE(
    vae,
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity,
    learning_rate=0.001)
  
  ptsne = VPTSNE(
    vae,
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity,
    learning_rate=0.001)

  def get_logger(loss_file, trustworthiness_file, knn_file):
    def log_fn(args):
      print(args)
      if isinstance(args[0], VAE):
        return
      loss_file.write(str(args[2]) + "\n")
      if args[1] % 400 == 0:
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
    "n_iters": 2001,
    "batch_size": batch_size,
    "deterministic": True,
    "fit_vae": False}
  fit_params_nondeterministic = {
    "n_iters": 2001,
    "batch_size": batch_size,
    "deterministic": False,
    "fit_vae": False}

  vptsne_log_files = [open("deterministic_output/%s_vptsne_%s.log" % (to_log, info), "w") for to_log in ["loss", "trustworthiness", "knn"]]
  ptsne_log_files = [open("deterministic_output/%s_ptsne_%s.log" % (to_log, info), "w") for to_log in ["loss", "trustworthiness", "knn"]]

  vptsne.fit(mnist_train_images, hook_fn=get_logger(*vptsne_log_files), **fit_params)
  ptsne.fit(mnist_train_images, hook_fn=get_logger(*ptsne_log_files), **fit_params_nondeterministic)

  vptsne.save_weights("models/mnist_vptsne_deterministic_%s.ckpt" % info, "models/mnist_vae_deterministic_%s.ckpt" % info)
  ptsne.save_weights("models/mnist_vptsne_nondeterministic_%s.ckpt" % info)

  for f in vptsne_log_files:
    f.close()
  for f in ptsne_log_files:
    f.close()

if __name__ == "__main__":

  vae = VAE(
    [n_input_dimensions],
    get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
    gaussian_prior_supplier,
    gaussian_supplier,
    get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
    bernoulli_supplier)

  vae.fit(mnist_train_images, n_iters=10000, batch_size=1000, hook_fn=print)

  for perplexity in [30]:
    for batch_size in [5000, 800, 400, 200]:
      for run_id in range(20):
        run_training(vae, n_latent_dimensions, perplexity, batch_size, run_id)
    for batch_size in [5000, 800, 400, 200]:
      for run_id in range(20, 40):
        run_training(vae, n_latent_dimensions, perplexity, batch_size, run_id)

