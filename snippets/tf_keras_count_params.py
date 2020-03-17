import tensorflow as tf
import numpy as np

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (10, 10), padding='same',
                           name='L0_conv2d', input_shape=(100, 100, 1)),
    tf.keras.layers.Dense(10, name='L1_dense', use_bias=False),
    tf.keras.layers.Dense(1, name='L2_dense'),
    tf.keras.layers.BatchNormalization(name='L3_bn')
])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# L0_conv2d (Conv2D)           (None, 100, 100, 1)       101       
# _________________________________________________________________
# L1_dense (Dense)             (None, 100, 100, 10)      10        
# _________________________________________________________________
# L2_dense (Dense)             (None, 100, 100, 1)       11        
# _________________________________________________________________
# L3_bn (BatchNormalization)   (None, 100, 100, 1)       4         
# =================================================================
# Total params: 126
# Trainable params: 124
# Non-trainable params: 2
# _________________________________________________________________

print(model.count_params())
# 126

print(model.layers[0].count_params())
# 101

print(issubclass(tf.keras.Model, tf.keras.layers.Layer))
# True

print(type(model.layers[2].get_weights()))
# <class 'list'>

print(len(model.layers[2].get_weights()))
# 2

kernel_weights, bias = model.layers[2].get_weights()

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

k_size, b_size = [w.size for w in model.layers[2].get_weights()]

print(k_size)
# 10

print(b_size)
# 1

print(type(model.layers[2].weights))
# <class 'list'>

print(len(model.layers[2].weights))
# 2

kernel_weights, bias = model.layers[2].weights

print(kernel_weights)
# <tf.Variable 'L2_dense/kernel:0' shape=(10, 1) dtype=float32, numpy=
# array([[-0.45019907],
#        [ 0.3547594 ],
#        [-0.01801795],
#        [ 0.5543849 ],
#        [-0.13720274],
#        [-0.71705985],
#        [ 0.30951375],
#        [-0.19865453],
#        [ 0.11943179],
#        [ 0.5920785 ]], dtype=float32)>

print(bias)
# <tf.Variable 'L2_dense/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>

print(type(kernel_weights))
# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

print(issubclass(type(kernel_weights), tf.Variable))
# True

print(kernel_weights.name)
# L2_dense/kernel:0

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
    print(w.name, '\t', np.prod(w.shape))
# L0_conv2d/kernel:0 	 100
# L0_conv2d/bias:0 	 1
# L1_dense/kernel:0 	 10
# L2_dense/kernel:0 	 10
# L2_dense/bias:0 	 1
# L3_bn/gamma:0 	 1
# L3_bn/beta:0 	 1
# L3_bn/moving_mean:0 	 1
# L3_bn/moving_variance:0 	 1

print(type(model.get_weights()))
# <class 'list'>

print(len(model.get_weights()))
# 9

print(type(model.get_weights()[0]))
# <class 'numpy.ndarray'>

for w in model.get_weights():
    print(w.size)
# 100
# 1
# 10
# 10
# 1
# 1
# 1
# 1
# 1

model.layers[1].trainable = False

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# L0_conv2d (Conv2D)           (None, 100, 100, 1)       101       
# _________________________________________________________________
# L1_dense (Dense)             (None, 100, 100, 10)      10        
# _________________________________________________________________
# L2_dense (Dense)             (None, 100, 100, 1)       11        
# _________________________________________________________________
# L3_bn (BatchNormalization)   (None, 100, 100, 1)       4         
# =================================================================
# Total params: 126
# Trainable params: 114
# Non-trainable params: 12
# _________________________________________________________________

trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
print(trainable_params)
# 114

non_trainable_params = sum(np.prod(w.shape) for w in model.non_trainable_weights)
print(non_trainable_params)
# 12

print(model.layers[3].trainable)
# True

for w in model.layers[3].trainable_weights:
    print(w.name, '\t', np.prod(w.shape))
# L3_bn/gamma:0 	 1
# L3_bn/beta:0 	 1

for w in model.layers[3].non_trainable_weights:
    print(w.name, '\t', np.prod(w.shape))
# L3_bn/moving_mean:0 	 1
# L3_bn/moving_variance:0 	 1

# NG
trainable_params = sum(l.count_params() for l in model.layers if l.trainable)
print(trainable_params)
# 116

trainable_params = sum(l.count_params() for l in model.layers if not l.trainable)
print(trainable_params)
# 10
