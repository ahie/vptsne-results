import numpy as np
import tensorflow as tf
import mnist
from vptsne.helpers import *

np.random.seed(0)
color_palette = np.random.rand(100, 3)
mnist_train_images, mnist_train_labels = mnist.train_images().reshape(60000, 784) / 255, mnist.train_labels()
mnist_test_images, mnist_test_labels = mnist.test_images().reshape(10000, 784) / 255, mnist.test_labels()

vae_layer_definitions = [
  (256, tf.nn.relu),
  (128, tf.nn.relu),
  (32, tf.nn.relu)]
vae_encoder_layers = LayerDefinition.from_array(vae_layer_definitions)
vae_decoder_layers = LayerDefinition.from_array(reversed(vae_layer_definitions))

vptsne_layers = LayerDefinition.from_array([
  (200, tf.nn.relu),
  (200, tf.nn.relu),
  (2000, tf.nn.relu),
  (2, None)])

