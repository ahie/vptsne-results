import numpy as np
import matplotlib.pyplot as plt
import hdata
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from vptsne import (VAE, PTSNE, VPTSNE)
from vptsne.helpers import *
from common import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.stats import pearsonr

all_patients = [
"H011_r2Asc",
"H024_rAsc2",
"H086_pAsc1",
"H087_iAsc1",
"H087_pAsc1",
"H089_iOme1",
"H092_pAsc2",
"H094_iAsc1",
"H094_iMes1",
"H094_pAsc1",
#"H100p_asc",
#"H116_pAsc",
"H116_pOme",
"H122_pAsc1",
"H131_iAsc1",
"H131_pAsc1",
"M087_rAsc",
"OC005_r4Asc",
"OC023_r2Asc"]

#train_data, _ = hdata.load_all()
train_data, _ = hdata.load_files([patient + "_log.csv" for patient in all_patients])
#scaler = StandardScaler().fit(train_data)
#train_data = scaler.transform(train_data)

patients = [
  #"H087_iAsc1",
  #"H087_pAsc1",
  "H100p_asc",
  "H116_pAsc"]
plot_data = []
for patient in patients:
  data, _ = hdata.load_files([patient + "_log.csv"])
  data = data[:10000]
#  data = scaler.transform(data)
  plot_data.append(data)

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
#  [train_data.shape[1]],
  vae,
  get_feed_forward_network_builder(vptsne_layers))

fit_params = {
  "hook_fn": print,
  "n_iters": 1500,
  "batch_size": 3000,
  "fit_vae": True,
  "n_vae_iters": 10000,
  "vae_batch_size": 1000}

vptsne.load_weights("models/vptsne_hdata_no_cd44_h100_116.ckpt", "models/vae_hdata_no_cd44_h100_116.ckpt")
#vptsne.fit(train_data, **fit_params)
#vptsne.save_weights("models/vptsne_hdata_no_cd44_h100_116.ckpt", "models/vae_hdata_no_cd44_h100_116.ckpt")

#all_data = {}
#all_data["patients"] = all_patients
#all_data["marker_order"] = hdata.marker_order
#for patient in all_patients:
#  print("Processing patient data", patient)
#  data, _ = hdata.load_files([patient + "_log.csv"])
#  data = data[:2500]
#  data = scaler.transform(data)
#  all_data[patient] = {}
#  for i, marker in enumerate(hdata.marker_order):
#    all_data[patient][marker] = data[:,i].reshape(-1).tolist()
#  all_data[patient]["x"] = vptsne.transform(data)[:,0].reshape(-1).tolist()
#  all_data[patient]["y"] = vptsne.transform(data)[:,1].reshape(-1).tolist()
#  all_data[patient]["scores"] = vae.score(data).tolist()
#
#import json
#with open("data.json", "w") as f:
#  json.dump(all_data, f)

class ExpressionLasso(object):
  def __init__(self, ax):
    self.lasso = LassoSelector(ax, onselect=self.onselect)
    self.fig, self.ax = plt.subplots(3, 5)
    self.ax = np.array(self.ax).reshape(-1)
  
  def onselect(self, verts):
    p = Path(verts)
    ind = p.contains_points(transformed)
    selected_data = concatenated_data[ind]
    mean_arr = np.zeros(len(hdata.marker_order))
    for i, marker in enumerate(hdata.marker_order):
      mean_arr[i] = np.mean(selected_data[:,i])
    for j, i in enumerate(mean_arr.argsort()[-15:][::-1]):
      self.ax[j].cla()
      self.ax[j].hist(selected_data[:,i])
      self.ax[j].set_title(hdata.marker_order[i])

  def disconnect(self):
    self.lasso.disconnect_events()

fig, ax = plt.subplots(1, 2)
concatenated_data = np.concatenate(plot_data)
transformed = vptsne.transform(concatenated_data)
threshold = -100.0

legend_handles = []
legend_titles = []
for patient, data in zip(patients, plot_data):
  t = vptsne.transform(data)
  h = ax[0].scatter(t[:,0], t[:,1], c=np.random.uniform(size=3), s=1.0)
  legend_handles.append(h)
  legend_titles.append(patient)
ax[0].legend(legend_handles, legend_titles, ncol=5, loc="lower left", bbox_to_anchor=(0,1.02,1,0.2), mode="expand")
elasso1 = ExpressionLasso(ax[0])

scores = vae.score(concatenated_data)
score_less = ax[1].scatter(transformed[:, 0][scores < threshold], transformed[:, 1][scores < threshold], s=1.00, c=[1,0,0])
score_greater = ax[1].scatter(transformed[:, 0][scores >= threshold], transformed[:, 1][scores >= threshold], s=1.00, c=[0,1,0])
ax[1].legend([score_less, score_greater], ["$\log p(\mathbf{x}) < -150$", "$\log p(\mathbf{x}) \geq -150$"], ncol=5, loc="lower left", bbox_to_anchor=(0,1.02,1,0.2), mode="expand")
elasso2 = ExpressionLasso(ax[1])

plt.show()

elasso2.disconnect()
elasso1.disconnect()

