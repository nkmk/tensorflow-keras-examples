import tensorflow as tf
import pprint

print(tf.__version__)
# 2.1.0

model = tf.keras.applications.VGG16(weights=None)

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

names = [l.name for l in model.layers]
pprint.pprint(names, compact=True)
# ['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1',
#  'block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3',
#  'block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
#  'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool', 'flatten',
#  'fc1', 'fc2', 'predictions']

pprint.pprint(model.layers)
# [<tensorflow.python.keras.engine.input_layer.InputLayer object at 0x131c7a650>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x104ce6590>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x131aba910>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x131c5cdd0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x12ab00e10>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x12aaf44d0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x12aaef890>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x131ebc290>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x131ec0bd0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x131ec6f50>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x131ec9e90>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x131ed8090>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x131edbdd0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x132a21dd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x132a26950>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x132a2bd10>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x132a35f90>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x132a3cbd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x132a41f10>,
#  <tensorflow.python.keras.layers.core.Flatten object at 0x132a46ed0>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x132a46f50>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x132a51710>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x132a59110>]

print(type(model.layers))
# <class 'list'>

print(model.layers[0])
# <tensorflow.python.keras.engine.input_layer.InputLayer object at 0x131c7a650>

print(type(model.layers[0]))
# <class 'tensorflow.python.keras.engine.input_layer.InputLayer'>

print(model.layers[0].name)
# input_1

names_pool = [l.name for l in model.layers if 'pool' in l.name]
print(names_pool)
# ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']

names_dense = [l.name for l in model.layers
               if isinstance(l, tf.keras.layers.Dense)]
print(names_dense)
# ['fc1', 'fc2', 'predictions']

names_dense_pool = [l.name for l in model.layers
                    if isinstance(l, (tf.keras.layers.Dense, tf.keras.layers.MaxPooling2D))]
pprint.pprint(names_dense_pool, compact=True)
# ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool',
#  'fc1', 'fc2', 'predictions']

names_not_conv = [l.name for l in model.layers
                  if not isinstance(l, tf.keras.layers.Conv2D)]
pprint.pprint(names_not_conv, compact=True)
# ['input_1', 'block1_pool', 'block2_pool', 'block3_pool', 'block4_pool',
#  'block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']

print(model.layers[0].name)
# input_1

print(model.layers[5].name)
# block2_conv2

print(model.layers[10].name)
# block3_pool

print(model.layers[-1].name)
# predictions

print(model.layers[-3].name)
# fc1

names_slice = [l.name for l in model.layers[5:10]]
print(names_slice)
# ['block2_conv2', 'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3']
