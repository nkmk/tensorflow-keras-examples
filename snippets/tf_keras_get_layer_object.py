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

l_block4_conv1 = model.get_layer('block4_conv1')
print(l_block4_conv1)
# <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d288090>

print(type(l_block4_conv1))
# <class 'tensorflow.python.keras.layers.convolutional.Conv2D'>

# print(model.get_layer('xxx'))
# ValueError: No such layer: xxx

l_11 = model.get_layer(index=11)
print(l_11)
# <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d288090>

print(type(l_11))
# <class 'tensorflow.python.keras.layers.convolutional.Conv2D'>

print(l_block4_conv1 is l_11)
# True

print(model.get_layer(index=-1).name)
# predictions

print(model.get_layer(index=-3).name)
# fc1

# print(model.get_layer(index=100))
# ValueError: Was asked to retrieve layer at index 100 but model only has 23 layers.

pprint.pprint(model.layers)
# [<tensorflow.python.keras.engine.input_layer.InputLayer object at 0x13d02a6d0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x1108bc0d0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13cff8c50>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x135eb4cd0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x135ea4c50>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x135e99c10>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d19ac90>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d26c290>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d270b90>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d276fd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d279e10>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d288090>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d28bd10>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13ddcfc90>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddd6fd0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13dddbed0>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13dde7e10>,
#  <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13ddecd90>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddf3e10>,
#  <tensorflow.python.keras.layers.core.Flatten object at 0x13ddf5c90>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13ddf5d10>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13de03910>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13de09210>]

print(type(model.layers))
# <class 'list'>

print(model.layers[11])
# <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x13d288090>

print(model.layers[11] is l_11)
# True

print(model.layers[-1].name)
# predictions

print(model.layers[-3].name)
# fc1

# print(model.layers[100])
# IndexError: list index out of range

l_1 = model.get_layer(index=1)
print(l_1)
# <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x1108bc0d0>

print(isinstance(l_1, tf.keras.layers.Layer))
# True

print(l_1.name)
# block1_conv1

print(l_1.count_params())
# 1792

print(l_1.trainable)
# True

l_1.trainable = False
print(l_1.trainable)
# False

# l_1.name = 'new_name'
# AttributeError: Can't set the attribute "name", likely because it conflicts with an existing read-only @property of the object. Please choose a different name.

l_1._name = 'new_name'
print(l_1.name)
# new_name

l_pool = [l for l in model.layers if 'pool' in l.name]
pprint.pprint(l_pool)
# [<tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x135eb4cd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d19ac90>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d279e10>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddd6fd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddf3e10>]

l_pool = [l for l in model.layers if isinstance(l, tf.keras.layers.MaxPooling2D)]
pprint.pprint(l_pool)
# [<tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x135eb4cd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d19ac90>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d279e10>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddd6fd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddf3e10>]

l_pool_dense = [l for l in model.layers
                if isinstance(l, (tf.keras.layers.MaxPooling2D, tf.keras.layers.Dense))]
pprint.pprint(l_pool_dense)
# [<tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x135eb4cd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d19ac90>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13d279e10>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddd6fd0>,
#  <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddf3e10>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13ddf5d10>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13de03910>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13de09210>]

l_tail = model.layers[-5:]
pprint.pprint(l_tail)
# [<tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x13ddf3e10>,
#  <tensorflow.python.keras.layers.core.Flatten object at 0x13ddf5c90>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13ddf5d10>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13de03910>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x13de09210>]

for l in model.layers:
    if isinstance(l, tf.keras.layers.Dense):
        l.trainable = False

for l in model.layers[-5:]:
    l.trainable = False
