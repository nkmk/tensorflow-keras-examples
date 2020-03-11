# Fails in TensorFlow 2.1.0. Please use tf-nightly or the next stable version.
# https://github.com/tensorflow/tensorflow/issues/35413

import tensorflow as tf
import pprint

print(tf.__version__)
# 2.2.0-dev20200309

inputs = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(inputs)
x = tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input)(x)

model = tf.keras.applications.vgg16.VGG16(input_tensor=x)

img_pil = tf.keras.preprocessing.image.load_img('../data/img/src/baboon.jpg')
img = tf.keras.preprocessing.image.img_to_array(img_pil)[tf.newaxis, ...]
print(img.shape)
# (1, 512, 512, 3)

pprint.pprint(tf.keras.applications.vgg16.decode_predictions(model.predict(img), top=5))
# [[('n02486410', 'baboon', 0.9816024),
#   ('n02484975', 'guenon', 0.007312194),
#   ('n02486261', 'patas', 0.0072130407),
#   ('n02487347', 'macaque', 0.0026990667),
#   ('n02493509', 'titi', 0.00031297794)]]

model_org = tf.keras.applications.vgg16.VGG16()

img2 = tf.image.resize(img, (224, 224))
img2 = tf.keras.applications.vgg16.preprocess_input(img2)
print(img2.shape)
# (1, 224, 224, 3)

pprint.pprint(tf.keras.applications.vgg16.decode_predictions(model_org.predict(img2), top=5))
# [[('n02486410', 'baboon', 0.9816024),
#   ('n02484975', 'guenon', 0.007312194),
#   ('n02486261', 'patas', 0.0072130407),
#   ('n02487347', 'macaque', 0.0026990667),
#   ('n02493509', 'titi', 0.00031297794)]]
