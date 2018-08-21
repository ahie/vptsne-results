import numpy as np
  
marker_order = [
  #                   "CD44",
  #                      |
  #                      v
  "CD90", "CD8", "Ki-67", "MUC1", "CD147", "E-CADH",
  "Cleaved-PARP", "N-CADH", "pS6", "CD117", "CA125", "CD166", "Sox2", "EpCAM",
  "ALDH", "CD45", "pAkt", "CD3", "PDL-1", "CD24", "pERK1-2", "PDL-2", "HE4", "PD1", "CD133"
]

def prep_holdout(data, labels, keep_percentage, folder, filename_prefix):
  to_shuffle = list(zip(data, labels))
  np.random.shuffle(to_shuffle)
  data, labels = zip(*to_shuffle)
  data = np.array(data)
  labels = np.array(labels)

  n_keep = int(data.shape[0] * keep_percentage)
  holdout = data[n_keep:]
  non_holdout = data[:n_keep]
  holdout_labels = labels[n_keep:]
  non_holdout_labels = labels[:n_keep]

  np.savetxt("%s/%s_holdout.tsv" % (folder, filename_prefix), holdout, delimiter="\t")
  np.savetxt("%s/%s_non_holdout.tsv" % (folder, filename_prefix), non_holdout, delimiter="\t")
  np.savetxt("%s/%s_holdout_labels.tsv" % (folder, filename_prefix), holdout_labels, delimiter="\t")
  np.savetxt("%s/%s_non_holdout_labels.tsv" % (folder, filename_prefix), non_holdout_labels, delimiter="\t")

  return holdout, non_holdout, holdout_labels, non_holdout_labels

def read_hdata(files, marker_order):
  all_data = np.empty((0, len(marker_order)))
  all_labels = np.empty((0, 1))

  def read_tsv(data_file):
    data = np.loadtxt(data_file, delimiter='\t', skiprows=1)
    with open(data_file, "r") as f:
      header = f.readline()
      header = header.strip().split()
      header = [h.strip('"') for h in header]
    return header, data

  for i, data_file in enumerate(files):
    header, data = read_tsv(data_file)

    column_order = []
    for marker in marker_order:
      marker_index = header.index(marker)
      column_order.append(marker_index)

    data = data[:, column_order]
    labels = [[i]] * data.shape[0]

    all_data = np.concatenate((all_data, data))
    all_labels = np.concatenate((all_labels, labels))

  return all_data, all_labels

def load_all(data_folder="CYTOMETRY_data/"):
  all_files = [
    "%s/H011_r2Asc_log.csv", #  0
    "%s/H024_rAsc2_log.csv", #  1
    "%s/H086_pAsc1_log.csv", #  2
    "%s/H087_iAsc1_log.csv", #  3
    "%s/H087_pAsc1_log.csv", #  4
    "%s/H089_iOme1_log.csv", #  5
    "%s/H092_pAsc2_log.csv", #  6
    "%s/H094_iAsc1_log.csv", #  7
    "%s/H094_iMes1_log.csv", #  8
    "%s/H094_pAsc1_log.csv", #  9
    "%s/H100p_asc_log.csv",  #  10
    "%s/H116_pAsc_log.csv",  #  11
    "%s/H116_pOme_log.csv",  #  12
    "%s/H122_pAsc1_log.csv", #  13
    "%s/H131_iAsc1_log.csv", #  14
    "%s/H131_pAsc1_log.csv", #  15
    "%s/M087_rAsc_log.csv",  #  16
    "%s/OC005_r4Asc_log.csv",#  17
    "%s/OC023_r2Asc_log.csv" #  18
  ]
  all_files = [file_path % data_folder for file_path in all_files]
  return read_hdata(all_files, marker_order)

def load_files(file_names, data_folder="CYTOMETRY_data/"):
  file_paths = [data_folder + file_name for file_name in file_names]
  return read_hdata(file_paths, marker_order)

