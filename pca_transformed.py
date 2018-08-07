import numpy as np
import matplotlib.pyplot as plt
import mnist
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.neighbors import KNeighborsClassifier as KNC

np.random.seed(0)
color_palette = np.random.rand(100, 3)
train_data, train_labels = mnist.train_images().reshape(60000, 784) / 255, mnist.train_labels()
test_data, test_labels = mnist.test_images().reshape(10000, 784) / 255, mnist.test_labels()

best = {"pca": 0, "vptsne": 0}

def save_best(identifier, score, transformed):
  global best
  if score > best[identifier]:
    best[identifier] = score
    plt.clf()
    for label in np.unique(train_labels):
      tmp = transformed[train_labels == label]
      plt.scatter(tmp[:, 0], tmp[:, 1], s=0.2, c=color_palette[label])
    plt.savefig("pca_transformed_output/%s_scatter.png" % identifier, format="png")
    plt.savefig("pca_transformed_output/%s_scatter.svg" % identifier, format="svg")
    np.save("pca_transformed_output/%s_scatter_data.npy" % identifier, transformed)

def run_training(n_principal_components, perplexity, batch_size, run_id):

  pca = PCA(n_components=n_principal_components).fit(train_data)

  vae = VAE(
    [train_data.shape[1]],
    get_gaussian_network_builder(vae_encoder_layers, n_principal_components),
    gaussian_prior_supplier,
    gaussian_supplier,
    get_bernoulli_network_builder(vae_decoder_layers, train_data.shape[1]),
    bernoulli_supplier)

  ptsne = PTSNE(
    [n_principal_components],
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity)

  vptsne = VPTSNE(
    vae,
    get_feed_forward_network_builder(vptsne_layers, batch_normalization=False),
    perplexity=perplexity)

  ptsne.fit(pca.transform(train_data), n_iters=1500, batch_size=batch_size)
  vptsne.fit(train_data, n_iters=1500, n_vae_iters=10000, batch_size=batch_size, vae_batch_size=1000)

  knn_score = KNC(n_neighbors=1).fit(
    ptsne.transform(pca.transform(train_data)), train_labels).score(
    ptsne.transform(pca.transform(test_data)), test_labels)
  knn_score_vptsne = KNC(n_neighbors=1).fit(
    vptsne.transform(train_data), train_labels).score(
    vptsne.transform(test_data), test_labels)

  tw = trustworthiness(
    test_data,
    ptsne.transform(pca.transform(test_data)),
    n_neighbors=12)
  tw_vptsne = trustworthiness(
    test_data,
    vptsne.transform(test_data),
    n_neighbors=12)

  save_best("pca", knn_score, ptsne.transform(pca.transform(train_data)))
  save_best("vptsne", knn_score_vptsne, vptsne.transform(train_data))

  return knn_score, tw, knn_score_vptsne, tw_vptsne

if __name__ == "__main__":
  for n_principal_components in [5, 10, 20, 30, 50, 40, 100, 3]:
    for run_id in range(5):
      res = run_training(n_principal_components, 30, 200, run_id)
      with open("pca_transformed_output/pca_knn_%s.log" % n_principal_components, "a") as f:
        f.write(str(res[0]) + "\n")
      with open("pca_transformed_output/pca_trustworthiness_%s.log" % n_principal_components, "a") as f:
        f.write(str(res[1]) + "\n")
      with open("pca_transformed_output/vptsne_knn_%s.log" % n_principal_components, "a") as f:
        f.write(str(res[2]) + "\n")
      with open("pca_transformed_output/vptsne_trustworthiness_%s.log" % n_principal_components, "a") as f:
        f.write(str(res[3]) + "\n")

