import tensorflow as tf
import pprint

print(tf.__version__)
# 2.1.0

model = tf.keras.applications.vgg16.VGG16(weights='imagenet')

model.summary()
# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0         
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544 
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312  
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000   
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

print(tf.keras.applications.vgg16.VGG16 is tf.keras.applications.VGG16)
# True

from tensorflow.keras.applications.vgg16 import VGG16

print(tf.keras.applications.vgg16.VGG16 is VGG16)
# True

img_pil = tf.keras.preprocessing.image.load_img(
    '../data/img/src/baboon.jpg', target_size=(224, 224)
)

print(type(img_pil))
# <class 'PIL.Image.Image'>

img = tf.keras.preprocessing.image.img_to_array(img_pil)
print(type(img))
# <class 'numpy.ndarray'>

print(img.shape)
# (224, 224, 3)

print(img.dtype)
# float32

print(img.min(), '-', img.max())
# 0.0 - 255.0

img = img[tf.newaxis, ...]
print(img.shape)
# (1, 224, 224, 3)

img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img)
print(img_preprocessed.min(), '-', img_preprocessed.max())
# -117.68 - 151.061

predict = model.predict(img_preprocessed)
print(type(predict))
# <class 'numpy.ndarray'>

print(predict.shape)
# (1, 1000)

print(predict[0][:10])
# [4.5425102e-07 6.2950056e-07 2.4502340e-09 1.5132209e-09 1.7509529e-09
#  1.2035696e-07 2.0865437e-08 1.2301771e-04 9.0907934e-06 5.3701660e-04]

result = tf.keras.applications.vgg16.decode_predictions(predict, top=5)
pprint.pprint(result)
# [[('n02486410', 'baboon', 0.96402234),
#   ('n02484975', 'guenon', 0.013725309),
#   ('n02486261', 'patas', 0.012976606),
#   ('n02487347', 'macaque', 0.0034710427),
#   ('n02493509', 'titi', 0.0015007565)]]

print(type(result))
# <class 'list'>

print(type(result[0]))
# <class 'list'>

print(type(result[0][0]))
# <class 'tuple'>

print(result[0][0][1])
# baboon
