import numpy as np
import gzip
import re

TIROSH_EXPR_FILE = 'SC_RNA_DATA/tirosh-expr-full.tsv.gz'
TIROSH_WEIGHTS_FILE = 'SC_RNA_DATA/tirosh-weights.tsv.gz'
tirosh_qc_gene_list = ["ACTB", "B2M", "HPRT1", "PSMB2", "PSMB4", "PPIA", "PRPS1", "PRPS1L1", "PRPS2", "PRPSAP1", "PRPSAP2", "RPL10", "RPL10A", "RPL10L", "RPL11", "RPL12", "RPL13", "RPL14", "RPL15", "RPL17", "RPL18", "RPL19", "RPL21", "RPL22", "RPL22L1", "RPL23", "RPL24", "RPL26", "RPL27", "RPL28", "RPL29", "RPL3", "RPL30", "RPL32", "RPL34", "RPL35", "RPL36", "RPL37", "RPL38", "RPL39", "RPL39L", "RPL3L", "RPL4", "RPL41", "RPL5", "RPL6", "RPL7", "RPL7A", "RPL7L1", "RPL8", "RPL9", "RPLP0", "RPLP1", "RPLP2", "RPS10", "RPS11", "RPS12", "RPS13", "RPS14", "RPS15", "RPS15A", "RPS16", "RPS17", "RPS18", "RPS19", "RPS20", "RPS21", "RPS24", "RPS25", "RPS26", "RPS27", "RPS27A", "RPS27L", "RPS28", "RPS29", "RPS3", "RPS3A", "RPS4X", "RPS5", "RPS6", "RPS6KA1", "RPS6KA2", "RPS6KA3", "RPS6KA4", "RPS6KA5", "RPS6KA6", "RPS6KB1", "RPS6KB2", "RPS6KC1", "RPS6KL1", "RPS7", "RPS8", "RPS9", "RPSA", "TRPS1", "UBB"]
tirosh_g1s_program = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"] 
tirosh_g2m_program = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]

PATEL_EXPR_FILE = 'SC_RNA_DATA/patel-expr-full.tsv.gz'
PATEL_WEIGHTS_FILE = 'SC_RNA_DATA/patel-weights.tsv.gz'

def _common_read(f, skip_columns=0):
  ret = []
  line = f.readline() # skip meta
  line = f.readline()
  while line:
    line = line.decode().split('\t')
    line = line[skip_columns:]
    ret.append([float(x) for x in line])
    line = f.readline()
  return np.array(ret)

def _get_data(fname, skip_columns=0):
  with gzip.open(fname, 'r') as f:
    return np.transpose(_common_read(f, skip_columns))

def _get_labels(fname, skip_columns=0):
  with gzip.open(fname, 'r') as f:
    return np.argmax(_common_read(f, skip_columns), axis=1)

def _get_tirosh_labels():
  with gzip.open(TIROSH_EXPR_FILE, 'r') as f:
    ids = f.readline().decode().split('\t')[2:]
    for i, id in enumerate(ids):
      if id[:6] == "monika":
        ids[i] = 1
      elif id[:2] == "SS":
        ids[i] = 2
      else:
        ids[i] = int(id[2:4])
    return np.array(ids)

def _get_tirosh_qc_indices():
  with gzip.open(TIROSH_EXPR_FILE, 'r') as f:
    f.readline()
    genes = []
    for line in f:
      genes.append(line.decode().split('\t')[0])
    qc_indices = [genes.index(qc_gene) for qc_gene in tirosh_qc_gene_list]
    g1s_indices = [genes.index(g1s_gene) for g1s_gene in tirosh_g1s_program]
    g2m_indices = [genes.index(g2m_gene) for g2m_gene in tirosh_g2m_program]
    return (qc_indices, g1s_indices, g2m_indices)

def _get_patel_labels():
  with gzip.open(PATEL_EXPR_FILE, 'r') as f:
    ids = f.readline().decode().split('\t')[2:]
    for i, id in enumerate(ids):
      m = re.search('MGH(\d+)', id)
      ids[i] = int(m.group(1))
    return np.array(ids)

def tirosh():
  return _get_data(TIROSH_EXPR_FILE, 2), _get_labels(TIROSH_WEIGHTS_FILE, 1), _get_tirosh_labels(), _get_tirosh_qc_indices()

def patel():
  return _get_data(PATEL_EXPR_FILE, 2), _get_labels(PATEL_WEIGHTS_FILE, 1), _get_patel_labels()

