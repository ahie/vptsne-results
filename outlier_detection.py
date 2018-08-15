import numpy as np
import matplotlib.pyplot as plt
import hdata
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox

train_data, _ = hdata.load_all()
outlier_data, _ = hdata.load_files(["H100p_asc_log.csv"])
inlier_data, _ = hdata.load_files(["H116_pAsc_log.csv"])

n_input_dimensions = train_data.shape[1]
n_latent_dimensions = 3

vae = VAE(
  [n_input_dimensions],
  get_gaussian_network_builder(vae_encoder_layers, n_latent_dimensions),
  gaussian_prior_supplier,
  gaussian_supplier,
  get_gaussian_network_builder(vae_decoder_layers, n_input_dimensions, constant_sigma=0.1),
  gaussian_supplier)

vptsne = VPTSNE(
  vae,
  get_feed_forward_network_builder(vptsne_layers))

fit_params = {
  "hook_fn": print,
  "n_iters": 2000,
  "batch_size": 2000,
  "fit_vae": True,
  "n_vae_iters": 12000,
  "vae_batch_size": 1000}

#vptsne.load_weights("models/vptsne_hdata.ckpt", "models/vae_hdata.ckpt")
vptsne.fit(train_data, **fit_params)
vptsne.save_weights("models/vptsne_hdata_2.ckpt", "models/vae_hdata_2.ckpt")

outlier_data = outlier_data[:20000]
inlier_data = inlier_data[:20000]

transformed_outlier = vptsne.transform(outlier_data)
transformed_inlier = vptsne.transform(inlier_data)
plt.subplot(121)
h100 = plt.scatter(transformed_outlier[:,0], transformed_outlier[:,1], c=[1,0,0], s=1.00)
h116 = plt.scatter(transformed_inlier[:,0], transformed_inlier[:,1], c=[0,1,0], s=1.00)
plt.legend([h100, h116], ["H100_pAsc", "H116_pAsc"], loc="upper right")

concatenated_data = np.concatenate((outlier_data, inlier_data))
transformed = vptsne.transform(concatenated_data)

#outlier_data = outlier_data[:2000]
#inlier_data = inlier_data[:2000]
#ous = vae.score(outlier_data)
#ins = vae.score(inlier_data)
#print(ous,ins, np.max(ous), np.max(ins))
#
#plt.clf()
#plt.subplot(121)
#plt.hist(ous)
#plt.subplot(122)
#plt.hist(ins)
#plt.show()
#raise Exception

scores = vae.score(concatenated_data)
plt.subplot(122)
threshold = -150.0
score_less = plt.scatter(transformed[:, 0][scores < threshold], transformed[:, 1][scores < threshold], s=1.00, c=[1,0,0])
score_greater = plt.scatter(transformed[:, 0][scores >= threshold], transformed[:, 1][scores >= threshold], s=1.00, c=[0,1,0])
plt.legend([score_less, score_greater], ["$\log p(\mathbf{x}) < -150$", "$\log p(\mathbf{x}) \geq -150$"], loc="upper right")

plt.show()

