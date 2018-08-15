import numpy as np
import matplotlib.pyplot as plt
import mnist
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from sklearn.manifold.t_sne import trustworthiness
from sklearn.neighbors import KNeighborsClassifier as KNC

np.random.seed(0)
color_palette = np.random.rand(100, 3)
train_data, train_labels = mnist.train_images().reshape(60000, 784) / 255, mnist.train_labels()
test_data, test_labels = mnist.test_images().reshape(10000, 784) / 255, mnist.test_labels()
n_input_dimensions = train_data.shape[1]
n_latent_dimensions = 5

non_corrupted_train_data = np.copy(train_data)
non_corrupted_test_data = np.copy(test_data)

best = {"ptsne": 0, "vptsne": 0}

def save_best(identifier, score, transformed):
  global best
  if score > best[identifier]:
    best[identifier] = score
    plt.clf()
    for label in np.unique(train_labels):
      tmp = transformed[train_labels == label]
      plt.scatter(tmp[:, 0], tmp[:, 1], s=0.2, c=color_palette[label])
    plt.savefig("corrupted_output/%s_scatter.png" % identifier, format="png")
    plt.savefig("corrupted_output/%s_scatter.svg" % identifier, format="svg")
    np.save("corrupted_output/%s_scatter_data.npy" % identifier, transformed)

def run_training(corruption_chance, perplexity, batch_size, run_id):

  global train_data, test_data
  corrupt = lambda x: 0 if np.random.uniform() <= corruption_chance else x
  train_data = np.vectorize(corrupt)(train_data)
  test_data = np.vectorize(corrupt)(test_data)

  vae = VAE(
    [n_input_dimensions],
    get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
    gaussian_prior_supplier,
    gaussian_supplier,
    get_bernoulli_network_builder(vae_decoder_layers, n_input_dimensions),
    bernoulli_supplier)

  ptsne = PTSNE(
    [n_input_dimensions],
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity)

  vptsne = VPTSNE(
    vae,
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity)

  ptsne.fit(train_data, n_iters=1500, batch_size=batch_size)
  vptsne.fit(train_data, n_iters=1500, n_vae_iters=10000, batch_size=batch_size, vae_batch_size=1000)

  knn_score = KNC(n_neighbors=1).fit(
    ptsne.transform(train_data), train_labels).score(
    ptsne.transform(test_data), test_labels)
  knn_score_vptsne = KNC(n_neighbors=1).fit(
    vptsne.transform(train_data), train_labels).score(
    vptsne.transform(test_data), test_labels)

  tw = trustworthiness(
    test_data,
    ptsne.transform(test_data),
    n_neighbors=12)
  tw_vptsne = trustworthiness(
    test_data,
    vptsne.transform(test_data),
    n_neighbors=12)

  save_best("ptsne", knn_score, ptsne.transform(non_corrupted_train_data))
  save_best("vptsne", knn_score_vptsne, vptsne.transform(non_corrupted_train_data))

  train_data = np.copy(non_corrupted_train_data)
  test_data = np.copy(non_corrupted_test_data)

  return knn_score, tw, knn_score_vptsne, tw_vptsne

if __name__ == "__main__":
  for corruption_chance in [0.2, 0.3, 0.4]:
    for run_id in range(3):
      res = run_training(corruption_chance, 30, 200, run_id)
      with open("corrupted_output/ptsne_knn_%s.log" % corruption_chance, "a") as f:
        f.write(str(res[0]) + "\n")
      with open("corrupted_output/ptsne_trustworthiness_%s.log" % corruption_chance, "a") as f:
        f.write(str(res[1]) + "\n")
      with open("corrupted_output/vptsne_knn_%s.log" % corruption_chance, "a") as f:
        f.write(str(res[2]) + "\n")
      with open("corrupted_output/vptsne_trustworthiness_%s.log" % corruption_chance, "a") as f:
        f.write(str(res[3]) + "\n")

