import tensorflow as tf
import numpy as np

print(tf.__version__)
# 2.1.0

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(name='bn', input_shape=(1,))
])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# bn (BatchNormalization)      (None, 1)                 4         
# =================================================================
# Total params: 4
# Trainable params: 2
# Non-trainable params: 2
# _________________________________________________________________

print(model.layers[0].trainable)
# True

for w in model.trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn/gamma:0                    [1.]
# bn/beta:0                     [0.]

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn/moving_mean:0              [0.]
# bn/moving_variance:0          [1.]

model.layers[0].trainable = False

print(model.layers[0].trainable_weights)
# []

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn/gamma:0                    [1.]
# bn/beta:0                     [0.]
# bn/moving_mean:0              [0.]
# bn/moving_variance:0          [1.]

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(name='bn', input_shape=(1,))
])

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_1/moving_mean:0            [0.]
# bn_1/moving_variance:0        [1.]

a = np.array([[100]]).astype('float32')
print(a)
# [[100.]]

print(a.shape)
# (1, 1)

print(model(a, training=True))
# tf.Tensor([[0.]], shape=(1, 1), dtype=float32)

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_1/moving_mean:0            [1.]
# bn_1/moving_variance:0        [0.99]

for i in range(1000):
    model(a, training=True)

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_1/moving_mean:0            [99.99573]
# bn_1/moving_variance:0        [4.273953e-05]

print(model(a, training=True))
# tf.Tensor([[0.]], shape=(1, 1), dtype=float32)

print(model(a, training=False))
# tf.Tensor([[0.13110352]], shape=(1, 1), dtype=float32)

print(model.predict(a))
# [[0.13110352]]

model.layers[0].trainable = False

print(model(a, training=True))
# tf.Tensor([[0.13110352]], shape=(1, 1), dtype=float32)

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(name='bn', input_shape=(1,))
])

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_2/moving_mean:0            [0.]
# bn_2/moving_variance:0        [1.]

for i in range(1000):
    model(a, training=False)

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_2/moving_mean:0            [0.]
# bn_2/moving_variance:0        [1.]

print(model(a, training=False))
# tf.Tensor([[99.95004]], shape=(1, 1), dtype=float32)

print(model.predict(a))
# [[99.95004]]

print((100 - 0) / np.sqrt(1 + 0.001))
# 99.95003746877732

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(name='bn', input_shape=(1,))
])

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/moving_mean:0            [0.]
# bn_3/moving_variance:0        [1.]

model.layers[0].trainable = False

for i in range(1000):
    model(a, training=True)

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/gamma:0                  [1.]
# bn_3/beta:0                   [0.]
# bn_3/moving_mean:0            [0.]
# bn_3/moving_variance:0        [1.]

print(model(a, training=True))
# tf.Tensor([[99.95004]], shape=(1, 1), dtype=float32)

print(model(a, training=False))
# tf.Tensor([[99.95004]], shape=(1, 1), dtype=float32)

print(model.predict(a))
# [[99.95004]]

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(a, a, verbose=0)
# WARNING:tensorflow:The list of trainable weights is empty. Make sure that you are not setting model.trainable to False before compiling the model.
# 
# <tensorflow.python.keras.callbacks.History at 0x13c3e78d0>

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/gamma:0                  [1.]
# bn_3/beta:0                   [0.]
# bn_3/moving_mean:0            [0.]
# bn_3/moving_variance:0        [1.]

model.layers[0].trainable = True

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(a, a, verbose=0)
# <tensorflow.python.keras.callbacks.History at 0x13c3d1e50>

for w in model.trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/gamma:0                  [1.]
# bn_3/beta:0                   [0.001]

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/moving_mean:0            [1.]
# bn_3/moving_variance:0        [0.99]

model.fit(a, a, epochs=1000, verbose=0)
# <tensorflow.python.keras.callbacks.History at 0x13c5625d0>

for w in model.trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/gamma:0                  [1.]
# bn_3/beta:0                   [0.9988577]

for w in model.non_trainable_weights:
    print('{:<30}{}'.format(w.name, w.numpy()))
# bn_3/moving_mean:0            [99.99573]
# bn_3/moving_variance:0        [4.273953e-05]
