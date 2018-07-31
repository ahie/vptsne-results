import os
import numpy as np
import gzip
import hashlib
import urllib.request

def download_fashion_mnist_file(directory, file_name):
  base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
  urllib.request.urlretrieve(base_url + file_name, os.path.join(directory, file_name))

def load_file(directory, file_name, checksum):
  if not os.path.exists(directory):
    os.makedirs(directory)
  path = os.path.join(directory, file_name)
  if not os.path.isfile(path):
    download_fashion_mnist_file(directory, file_name)
  if hashlib.md5(open(path, "rb").read()).hexdigest() != checksum:
    raise Exception("Fashion-MNIST file " + path + " has incorrect checksum")
  with gzip.open(path, "rb") as f:
    return np.frombuffer(f.read(), dtype="uint8")

def load_images(image_data):
  return image_data[16:].reshape([-1, 28, 28, 1]) / 255

def load_labels(label_data):
  return label_data[8:]

def train(directory):
  image_data = load_file(directory, "train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d")
  label_data = load_file(directory, "train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe")
  return load_images(image_data), load_labels(label_data)

def test(directory):
  image_data = load_file(directory, "t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79")
  label_data = load_file(directory, "t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
  return load_images(image_data), load_labels(label_data)

