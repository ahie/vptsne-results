import mnist
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
mnist_train_images, mnist_train_labels = mnist.train_images().reshape(60000, 784) / 255, mnist.train_labels()
mnist_test_images, mnist_test_labels = mnist.test_images().reshape(10000, 784) / 255, mnist.test_labels()
n_input_dimensions = mnist_train_images.shape[1]

def run_training(n_latent_dimensions, perplexity, batch_size, percent_missing, run_id):

  data_points = mnist_train_images.shape[0]
  indices = np.random.choice(data_points, int(data_points * (1 - percent_missing)), replace=False)
  train_data = mnist_train_images[indices]
  train_labels = mnist_train_labels[indices]
  test_data = mnist_test_images
  test_labels = mnist_test_labels

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

  def hook(args):
    print(args)
    if np.isnan(args[2]):
      raise Exception

  fit_params = {
    "hook_fn": hook,
    "n_iters": 1500,
    "batch_size": batch_size,
    "deterministic": True,
    "fit_vae": True,
    "n_vae_iters": 10000,
    "vae_batch_size": 300}

  vptsne.fit(train_data, **fit_params)
  ptsne.fit(train_data, **fit_params)

  vptsne_knn_score = KNC(n_neighbors=1).fit(vptsne.transform(train_data), train_labels).score(vptsne.transform(test_data), test_labels)
  ptsne_knn_score = KNC(n_neighbors=1).fit(ptsne.transform(train_data), train_labels).score(ptsne.transform(test_data), test_labels)

  vptsne_trustworthiness = trustworthiness(test_data, vptsne.transform(test_data), n_neighbors=12)
  ptsne_trustworthiness = trustworthiness(test_data, ptsne.transform(test_data), n_neighbors=12)

  return vptsne_knn_score, ptsne_knn_score, vptsne_trustworthiness, ptsne_trustworthiness

if __name__ == "__main__":
  for percent_missing in [0.5, 0.8, 0.9, 0.6, 0.3, 0.2, 0.7, 0.95, 0.4, 0.99]:
    for run_id in range(10):
      while True:
        try:
          res = run_training(3, 30, 200, percent_missing, run_id)
          break
        except:
          pass
      with open("missing_data_output/vptsne_subset_knn_score_%s.log" % percent_missing, "a") as f:
        f.write(str(res[0]) + "\n")
      with open("missing_data_output/ptsne_subset_knn_score_%s.log" % percent_missing, "a") as f:
        f.write(str(res[1]) + "\n")
      with open("missing_data_output/vptsne_subset_trustworthiness_%s.log" % percent_missing, "a") as f:
        f.write(str(res[2]) + "\n")
      with open("missing_data_output/ptsne_subset_trustworthiness_%s.log" % percent_missing, "a") as f:
        f.write(str(res[3]) + "\n")

