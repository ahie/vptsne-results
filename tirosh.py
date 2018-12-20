import time
import numpy as np
import tensorflow as tf
import tensorflow.distributions as tfds
import matplotlib.pyplot as plt
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from umap import UMAP
from scdata import tirosh
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import trustworthiness
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier as KNC

curr_millis = lambda: int(round(time.time() * 1000))

np.random.seed(0)
color_palette = np.random.rand(100, 3)

tirosh_data, tirosh_cell_type_labels, tirosh_labels, gene_indices = tirosh()
tirosh_data_orig = np.copy(tirosh_data) # np.log2(np.copy(tirosh_data) / 10 + 1)
tirosh_data_relative_expression = tirosh_data_orig - np.mean(tirosh_data_orig, axis=0)
#tirosh_data = np.sqrt(tirosh_data)
#from sklearn.decomposition import PCA
#tirosh_data = PCA(n_components=500).fit_transform(tirosh_data)

n_input_dimensions = tirosh_data.shape[1]
n_latent_dimensions = 50

_, g1s_indices, g2m_indices = gene_indices

#qc = np.logical_and(np.mean(tirosh_data_orig[:,qc_gene_indices], axis=1) > 3, np.count_nonzero(tirosh_data_orig, axis=1) > 1700)
#
#tirosh_data = tirosh_data[qc]
#tirosh_cell_type_labels = tirosh_cell_type_labels[qc]
#tirosh_labels = tirosh_labels[qc]
#tirosh_data_orig = tirosh_data_orig[qc]
#tirosh_data_relative_expression = tirosh_data_relative_expression[qc]

vae_layer_definitions = [
  (1000, tf.nn.relu),
  (1000, tf.nn.relu),
  (500, tf.nn.relu)]
vae_encoder_layers = LayerDefinition.from_array(vae_layer_definitions)
vae_decoder_layers = LayerDefinition.from_array(reversed(vae_layer_definitions))

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.01, output_hidden=True),
  gaussian_supplier)

fit_params = {
  "hook_fn": print,
  "n_iters": 2000,
  "batch_size": 500,
  "deterministic": False,
  "fit_vae": True,
  "n_vae_iters": 10000,
  "vae_batch_size": 500}

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=True),
  perplexity=10)

ptsne = PTSNE(
  [n_input_dimensions],
  get_feed_forward_network_builder(vptsne_layers, batch_normalization=True),
  perplexity=10)

pca = PCA(n_components=2)
umap = UMAP(n_components=2, verbose=True)
tsne = TSNE(n_components=2, perplexity=30)

estimators = [tsne]#[tsne, pca]#[vptsne, ptsne, vae, pca, umap, tsne]

def fit_transform_fn(estimator):
  print("Running fit_transform with estimator", estimator.__class__.__name__)
  start = curr_millis()
  if isinstance(estimator, PTSNE):
    transformed = estimator.fit_transform(tirosh_data, **fit_params)
  elif isinstance(estimator, VAE): # Already fitted
    transformed = estimator.transform(tirosh_data)
  else:
    transformed = estimator.fit_transform(tirosh_data)
  print(estimator.__class__.__name__, "fit_transform completed in", curr_millis() - start, "(ms)")
  return transformed


transformed_all = [fit_transform_fn(estimator) for estimator in estimators]
#vptsne.save_weights("models/tirosh_vptsne_tmp.ckpt", "models/tirosh_vae_tmp.ckpt")
#vptsne.load_weights("models/tirosh_vptsne_2.ckpt", "models/tirosh_vae_2.ckpt")
#transformed_all = [vptsne.transform(tirosh_data)]
transformed = transformed_all[0]
#transformed = np.loadtxt("tirosh_vptsne.tsv", delimiter="\t")[:,[0,1]]
#transformed_all = [transformed]

from sklearn.preprocessing import MinMaxScaler as mms
from scipy.stats import pearsonr
from scipy.stats import spearmanr

mitf = np.zeros((transformed.shape[0], 3))
axl = np.zeros((transformed.shape[0], 3))

bins = np.argsort(np.sum(tirosh_data_orig[tirosh_cell_type_labels == 0], axis=0))
bin_size = int(tirosh_data_orig.shape[1] / 25)

def get_control_genes(for_gene):
  idx = np.where(bins == for_gene)[0][0]
  bin = int(idx / bin_size)
  return np.random.choice(bins[bin * bin_size : (bin + 1) * bin_size], 100, replace=False)

def control(genes_to_control, gene_expression_data):
  final_cell_scores = np.copy(gene_expression_data)
  control = []
  for gene_to_control in genes_to_control:
    control.extend(get_control_genes(gene_to_control))
  return np.mean(final_cell_scores[:,genes_to_control], axis=1) - np.mean(final_cell_scores[:,control], axis=1)

mitf_index = 8392
#axl_index = 11968

mitf_corr = np.zeros(tirosh_data_orig.shape[1])
malignant = tirosh_data_orig[tirosh_cell_type_labels == 0]
for i in range(tirosh_data_orig.shape[1]):
  mitf_corr[i], _ = pearsonr(malignant[:,i], malignant[:,mitf_index])
mitf_corr[np.isnan(mitf_corr)] = -np.inf # many genes have 0 expression for all cells, resulting in nan correlations
mitf_program_gene_indices = np.argsort(mitf_corr)[::-1][:100]
mitf_cell_scores = control(mitf_program_gene_indices, tirosh_data_relative_expression)

axl_corr = np.zeros(tirosh_data_orig.shape[1])
for i in range(tirosh_data_orig.shape[1]):
  axl_corr[i], _ = pearsonr(malignant[:,i], mitf_cell_scores[tirosh_cell_type_labels == 0])
axl_corr[np.isnan(axl_corr)] = np.inf
axl_program_gene_indices = np.argsort(axl_corr)[:100]
axl_cell_scores = control(axl_program_gene_indices, tirosh_data_relative_expression)

#mel = axl_cell_scores[np.logical_and(tirosh_labels == 81, tirosh_cell_type_labels == 0)]
#plt.hist(mel)
#plt.show()
#mel = mitf_cell_scores[np.logical_and(tirosh_labels == 81, tirosh_cell_type_labels == 0)]
#plt.hist(mel)
#plt.show()

mitf[:,0] = np.clip(mms().fit_transform(mitf_cell_scores.reshape(-1,1)).reshape(-1), 0, 1)
axl[:,0] = np.clip(mms().fit_transform(axl_cell_scores.reshape(-1,1)).reshape(-1), 0, 1)

#for tumor in [53, 81, 82, 79, 80, 59, 84, 78, 88, 71]:
#  m = np.mean(mitf_cell_scores[np.logical_and(tirosh_labels == tumor, tirosh_cell_type_labels == 0)])
#  a = np.mean(axl_cell_scores[np.logical_and(tirosh_labels == tumor, tirosh_cell_type_labels == 0)])
#  plt.scatter(m, a)
#  plt.annotate("Mel" + str(tumor), (m, a))
#plt.show()
#
#plt.subplot(211)
#plt.hist(mitf_cell_scores)
#plt.subplot(212)
#plt.hist(axl_cell_scores)
#plt.show()

g1s_scores = np.mean(tirosh_data_relative_expression[:,g1s_indices], axis=1)
g2m_scores = np.mean(tirosh_data_relative_expression[:,g2m_indices], axis=1)
m = np.max(np.column_stack((g1s_scores, g2m_scores)), axis=1)
am = np.argmax(np.column_stack((g1s_scores, g2m_scores)), axis=1)
g1s = np.logical_and(g1s_scores > 1, am == 0)
g2m = np.logical_and(g2m_scores > 1, am == 1)

cycling = m > 1
non_cycling = np.logical_not(cycling)

cycle_gradient = np.zeros((tirosh_data[cycling].shape[0], 3))
cycle_gradient[:,0] = mms().fit_transform(m[cycling].reshape(-1,1)).reshape(-1)

cycle = np.zeros(tirosh_data.shape[0])
cycle[cycling] = cycle_gradient[:,0]

"""
#plt.subplot(325)
t = transformed[non_cycling]
noncyc = plt.scatter(t[:,0], t[:,1], color="green", s=50)
t = transformed[cycling]
cyc = plt.scatter(t[:,0], t[:,1], color=cycle_gradient, s=50)
plt.legend([cyc, noncyc], ["Cycling", "Non-cycling"], loc='lower right')
plt.title("Cycling/Non-cycling")

plt.show()
#plt.subplot(323)
plt.scatter(transformed[:,0], transformed[:,1], color=mitf, s=50)
plt.title("MITF")
plt.show()
#plt.subplot(324)
plt.scatter(transformed[:,0], transformed[:,1], color=axl, s=50)
plt.title("AXL")

for i, transformed in enumerate(transformed_all):
  legend_handles = []
  legend_labels = []
  plt.show()
  #plt.subplot(322)
  for label in np.unique(tirosh_labels):
    tmp = transformed[tirosh_labels == label]
    h = plt.scatter(tmp[:, 0], tmp[:, 1], s=50, c=color_palette[label])
    if label != 1 and label != 2:
      legend_handles.append(h)
      legend_labels.append("Mel" + str(label))
  plt.legend(legend_handles, legend_labels, loc='lower right', ncol=4)#, bbox_to_anchor=(0,1.02,-0.5,0.2), ncol=5)
  plt.title("Tumor")

  plt.show()
  #plt.subplot(321)
  legend_handles = []
  legend_labels = []
  for label in np.unique(tirosh_cell_type_labels):
    tmp = transformed[tirosh_cell_type_labels == label]
    h = plt.scatter(tmp[:, 0], tmp[:, 1], s=50, c=color_palette[label])
    legend_handles.append(h)
  plt.title("Cell Type")
  plt.legend(legend_handles, ["Melanoma", "B", "T", "Macro", "Endo", "CAF", "NK"], loc='lower right')
  plt.show()
"""

data = np.zeros((tirosh_data_orig.shape[0], 7))
data[:,0] = transformed[:,0]
data[:,1] = transformed[:,1]
data[:,2] = tirosh_cell_type_labels
data[:,3] = tirosh_labels
data[:,4] = mitf[:,0]
data[:,5] = axl[:,0]
data[:,6] = m
np.savetxt('tirosh.tsv', data, delimiter='\t')

