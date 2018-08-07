import tensorflow as tf
from vptsne.helpers import *

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

