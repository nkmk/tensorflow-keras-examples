import tensorflow as tf
import pprint

print(tf.__version__)
# 2.1.0

model = tf.keras.applications.vgg16.VGG16(weights='imagenet')

img_pil = tf.keras.preprocessing.image.load_img(
    '../data/img/src/baboon.jpg', target_size=(224, 224)
)

img = tf.keras.applications.vgg16.preprocess_input(
    tf.keras.preprocessing.image.img_to_array(img_pil)[tf.newaxis]
)

predict = model.predict(img)

result = tf.keras.applications.vgg16.decode_predictions(predict, top=5)
pprint.pprint(result)
# [[('n02486410', 'baboon', 0.96402234),
#   ('n02484975', 'guenon', 0.013725309),
#   ('n02486261', 'patas', 0.012976606),
#   ('n02487347', 'macaque', 0.0034710427),
#   ('n02493509', 'titi', 0.0015007565)]]
