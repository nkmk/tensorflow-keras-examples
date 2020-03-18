import tensorflow as tf
import numpy as np

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), padding='same',
                           name='L0_conv2d', input_shape=(10, 10, 1)),
    tf.keras.layers.Flatten(name='L1_flatten'),
    tf.keras.layers.Dense(10, name='L2_dense', use_bias=False),
    tf.keras.layers.Dense(1, name='L3_dense'),
    tf.keras.layers.BatchNormalization(name='L4_bn')
])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# L0_conv2d (Conv2D)           (None, 10, 10, 1)         10        
# _________________________________________________________________
# L1_flatten (Flatten)         (None, 100)               0         
# _________________________________________________________________
# L2_dense (Dense)             (None, 10)                1000      
# _________________________________________________________________
# L3_dense (Dense)             (None, 1)                 11        
# _________________________________________________________________
# L4_bn (BatchNormalization)   (None, 1)                 4         
# =================================================================
# Total params: 1,025
# Trainable params: 1,023
# Non-trainable params: 2
# _________________________________________________________________

print(model.count_params())
# 1025

print(model.layers[0].count_params())
# 10

print(issubclass(tf.keras.Model, tf.keras.layers.Layer))
# True

print(type(model.layers[3].get_weights()))
# <class 'list'>

print(len(model.layers[3].get_weights()))
# 2

kernel_weights, bias = model.layers[3].get_weights()

print(kernel_weights)
# [[-0.45019907]
#  [ 0.3547594 ]
#  [-0.01801795]
#  [ 0.5543849 ]
#  [-0.13720274]
#  [-0.71705985]
#  [ 0.30951375]
#  [-0.19865453]
#  [ 0.11943179]
#  [ 0.5920785 ]]

print(type(kernel_weights))
# <class 'numpy.ndarray'>

print(bias)
# [0.]

print(type(bias))
# <class 'numpy.ndarray'>

print(kernel_weights.size)
# 10

print(bias.size)
# 1

k_size, b_size = [w.size for w in model.layers[3].get_weights()]

print(k_size)
# 10

print(b_size)
# 1

print(type(model.layers[3].weights))
# <class 'list'>

print(len(model.layers[3].weights))
# 2

kernel_weights, bias = model.layers[3].weights

print(type(kernel_weights))
# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

print(issubclass(type(kernel_weights), tf.Variable))
# True

print(kernel_weights.name)
# L3_dense/kernel:0

print(kernel_weights.shape)
# (10, 1)

print(kernel_weights.numpy())
# [[-0.45019907]
#  [ 0.3547594 ]
#  [-0.01801795]
#  [ 0.5543849 ]
#  [-0.13720274]
#  [-0.71705985]
#  [ 0.30951375]
#  [-0.19865453]
#  [ 0.11943179]
#  [ 0.5920785 ]]

print(kernel_weights.numpy().size)
# 10

print(np.prod(kernel_weights.shape))
# 10

print(type(model.weights))
# <class 'list'>

print(len(model.weights))
# 9

print(type(model.weights[0]))
# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

for w in model.weights:
    print('{:<25}{:<15}{}'.format(w.name, str(w.shape), np.prod(w.shape)))
# L0_conv2d/kernel:0       (3, 3, 1, 1)   9
# L0_conv2d/bias:0         (1,)           1
# L2_dense/kernel:0        (100, 10)      1000
# L3_dense/kernel:0        (10, 1)        10
# L3_dense/bias:0          (1,)           1
# L4_bn/gamma:0            (1,)           1
# L4_bn/beta:0             (1,)           1
# L4_bn/moving_mean:0      (1,)           1
# L4_bn/moving_variance:0  (1,)           1

print(type(model.get_weights()))
# <class 'list'>

print(len(model.get_weights()))
# 9

print(type(model.get_weights()[0]))
# <class 'numpy.ndarray'>

for w in model.get_weights():
    print(w.size)
# 9
# 1
# 1000
# 10
# 1
# 1
# 1
# 1
# 1

model.layers[2].trainable = False

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# L0_conv2d (Conv2D)           (None, 10, 10, 1)         10        
# _________________________________________________________________
# L1_flatten (Flatten)         (None, 100)               0         
# _________________________________________________________________
# L2_dense (Dense)             (None, 10)                1000      
# _________________________________________________________________
# L3_dense (Dense)             (None, 1)                 11        
# _________________________________________________________________
# L4_bn (BatchNormalization)   (None, 1)                 4         
# =================================================================
# Total params: 1,025
# Trainable params: 23
# Non-trainable params: 1,002
# _________________________________________________________________

print(model.layers[2].trainable)
# False

print(model.layers[2].trainable_weights)
# []

print(model.layers[2].non_trainable_weights == model.layers[2].weights)
# True

print(model.layers[3].trainable)
# True

print(model.layers[3].non_trainable_weights)
# []

print(model.layers[3].trainable_weights == model.layers[3].weights)
# True

trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
print(trainable_params)
# 23

non_trainable_params = sum(np.prod(w.shape) for w in model.non_trainable_weights)
print(non_trainable_params)
# 1002

print(model.layers[4].trainable)
# True

for w in model.layers[4].trainable_weights:
    print('{:<25}{}'.format(w.name, np.prod(w.shape)))
# L4_bn/gamma:0            1
# L4_bn/beta:0             1

for w in model.layers[4].non_trainable_weights:
    print('{:<25}{}'.format(w.name, np.prod(w.shape)))
# L4_bn/moving_mean:0      1
# L4_bn/moving_variance:0  1

# NG
trainable_params = sum(l.count_params() for l in model.layers if l.trainable)
print(trainable_params)
# 25

trainable_params = sum(l.count_params() for l in model.layers if not l.trainable)
print(trainable_params)
# 1000
