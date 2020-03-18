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

l3 = model.layers[3]

print(type(l3.get_weights()))
# <class 'list'>

print(len(l3.get_weights()))
# 2

print(l3.get_weights()[0])
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

print(type(l3.get_weights()[0]))
# <class 'numpy.ndarray'>

print(l3.get_weights()[1])
# [0.]

print(type(l3.get_weights()[1]))
# <class 'numpy.ndarray'>

print(len(model.layers[0].weights))
# 2

print(len(model.layers[1].weights))
# 0

print(len(model.layers[2].weights))
# 1

print(len(model.layers[3].weights))
# 2

print(len(model.layers[4].weights))
# 4

print(type(l3.weights))
# <class 'list'>

print(len(l3.weights))
# 2

print(l3.weights[0])
# <tf.Variable 'L3_dense/kernel:0' shape=(10, 1) dtype=float32, numpy=
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

print(type(l3.weights[0]))
# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

print(l3.weights[1])
# <tf.Variable 'L3_dense/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>

print(type(l3.weights[1]))
# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

print(issubclass(type(l3.weights[0]), tf.Variable))
# True

print(l3.weights[0].name)
# L3_dense/kernel:0

print(l3.weights[0].shape)
# (10, 1)

print(l3.weights[1].name)
# L3_dense/bias:0

print(l3.weights[1].shape)
# (1,)

print(l3.weights[0].numpy())
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

print(l3.weights[1].numpy())
# [0.]

print(np.array_equal(l3.weights[0].numpy(), l3.get_weights()[0]))
# True

print(np.array_equal(l3.weights[1].numpy(), l3.get_weights()[1]))
# True

for w in model.layers[0].weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L0_conv2d/kernel:0       (3, 3, 1, 1)
# L0_conv2d/bias:0         (1,)

print(model.layers[1].weights)
# []

for w in model.layers[2].weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L2_dense/kernel:0        (100, 10)

for w in model.layers[3].weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L3_dense/kernel:0        (10, 1)
# L3_dense/bias:0          (1,)

for w in model.layers[4].weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L4_bn/gamma:0            (1,)
# L4_bn/beta:0             (1,)
# L4_bn/moving_mean:0      (1,)
# L4_bn/moving_variance:0  (1,)

print(l3.weights == l3.variables)
# True

print(l3.trainable)
# True

print(l3.trainable_weights == l3.weights)
# True

print(l3.non_trainable_weights)
# []

l3.trainable = False

print(l3.non_trainable_weights == l3.weights)
# True

print(l3.trainable_weights)
# []

print(model.layers[4].trainable)
# True

for w in model.layers[4].trainable_weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L4_bn/gamma:0            (1,)
# L4_bn/beta:0             (1,)

for w in model.layers[4].non_trainable_weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L4_bn/moving_mean:0      (1,)
# L4_bn/moving_variance:0  (1,)

print(l3.kernel)
# <tf.Variable 'L3_dense/kernel:0' shape=(10, 1) dtype=float32, numpy=
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

print(l3.kernel is l3.weights[0])
# True

print(l3.bias)
# <tf.Variable 'L3_dense/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>

print(l3.bias is l3.weights[1])
# True

# print(model.layers[1].kernel)
# AttributeError: 'Flatten' object has no attribute 'kernel'

# print(model.layers[4].kernel)
# AttributeError: 'BatchNormalization' object has no attribute 'kernel'

print(model.layers[2].bias)
# None

print(model.layers[4].gamma)
# <tf.Variable 'L4_bn/gamma:0' shape=(1,) dtype=float32, numpy=array([1.], dtype=float32)>

print(model.layers[4].gamma is model.layers[4].weights[0])
# True

print(issubclass(tf.keras.Model, tf.keras.layers.Layer))
# True

print(type(model))
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>

print(issubclass(tf.keras.Sequential, tf.keras.Model))
# True

print(issubclass(tf.keras.Sequential, tf.keras.layers.Layer))
# True

print(type(model.weights))
# <class 'list'>

print(len(model.weights))
# 9

print(type(model.weights[0]))
# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

for w in model.weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L0_conv2d/kernel:0       (3, 3, 1, 1)
# L0_conv2d/bias:0         (1,)
# L2_dense/kernel:0        (100, 10)
# L3_dense/kernel:0        (10, 1)
# L3_dense/bias:0          (1,)
# L4_bn/gamma:0            (1,)
# L4_bn/beta:0             (1,)
# L4_bn/moving_mean:0      (1,)
# L4_bn/moving_variance:0  (1,)

# print(model.kernel)
# AttributeError: 'Sequential' object has no attribute 'kernel'

print(type(model.get_weights()))
# <class 'list'>

print(len(model.get_weights()))
# 9

print(type(model.get_weights()[0]))
# <class 'numpy.ndarray'>

for a in model.get_weights():
    print(a.shape)
# (3, 3, 1, 1)
# (1,)
# (100, 10)
# (10, 1)
# (1,)
# (1,)
# (1,)
# (1,)
# (1,)

inner_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, name='L_in_0', input_shape=(1000,)),
    tf.keras.layers.Dense(10, name='L_in_1')
], name='Inner_model')

outer_model = tf.keras.Sequential([
    inner_model,
    tf.keras.layers.Dense(1, name='L_out_1')
])

outer_model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Inner_model (Sequential)     (None, 10)                101110    
# _________________________________________________________________
# L_out_1 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

outer_model.layers[0].summary()
# Model: "Inner_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# L_in_0 (Dense)               (None, 100)               100100    
# _________________________________________________________________
# L_in_1 (Dense)               (None, 10)                1010      
# =================================================================
# Total params: 101,110
# Trainable params: 101,110
# Non-trainable params: 0
# _________________________________________________________________

print(len(outer_model.weights))
# 6

for w in outer_model.weights:
    print('{:<25}{}'.format(w.name, w.shape))
# L_in_0/kernel:0          (1000, 100)
# L_in_0/bias:0            (100,)
# L_in_1/kernel:0          (100, 10)
# L_in_1/bias:0            (10,)
# L_out_1/kernel:0         (10, 1)
# L_out_1/bias:0           (1,)

print(len(outer_model.get_weights()))
# 6

for a in outer_model.get_weights():
    print(a.shape)
# (1000, 100)
# (100,)
# (100, 10)
# (10,)
# (10, 1)
# (1,)
