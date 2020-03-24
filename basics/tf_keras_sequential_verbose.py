import tensorflow as tf
import numpy as np
from PIL import Image

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(type(x_train))
# <class 'numpy.ndarray'>

print(x_train.shape)
# (60000, 28, 28)

print(x_train.dtype)
# uint8

print(x_train.min(), '-', x_train.max())
# 0 - 255

Image.fromarray(x_train[0]).resize((256, 256)).save('../data/img/dst/mnist_sample_resize.png')

print(x_test.shape)
# (10000, 28, 28)

print(type(y_train))
# <class 'numpy.ndarray'>

print(y_train.shape)
# (60000,)

print(y_train.dtype)
# uint8

print(y_train.min(), '-', y_train.max())
# 0 - 9

print(np.unique(y_train))
# [0 1 2 3 4 5 6 7 8 9]

print(y_train[0])
# 5

print(y_test.shape)
# (10000,)

x_train = x_train / 255
x_test = x_test / 255

print(x_train.dtype)
# float64

print(x_train.min(), '-', x_train.max())
# 0.0 - 1.0

print(tf.keras.Sequential is tf.keras.models.Sequential)
# True

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
], name='my_model')

model.summary()
# Model: "my_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_layer (Flatten)      (None, 784)               0         
# _________________________________________________________________
# dense (Dense)                (None, 128)               100480    
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

model_1 = tf.keras.Sequential(name='my_model_1')
model_1.add(tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer_1'))
model_1.add(tf.keras.layers.Dense(128, activation='relu'))
model_1.add(tf.keras.layers.Dropout(0.2))
model_1.add(tf.keras.layers.Dense(10, activation='softmax'))

model_1.summary()
# Model: "my_model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_layer_1 (Flatten)    (None, 784)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               100480    
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 128)               0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

model_2 = Sequential([
    Flatten(input_shape=(28, 28), name='flatten_layer_2'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
], name='my_model_2')

model_2.summary()
# Model: "my_model_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_layer_2 (Flatten)    (None, 784)               0         
# _________________________________________________________________
# dense_4 (Dense)              (None, 128)               100480    
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0         
# _________________________________________________________________
# dense_5 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

print(tf.keras.Sequential is Sequential)
# True

import tensorflow.keras.layers as L

print(tf.keras.layers.Dense is L.Dense)
# True

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model_1.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=[tf.keras.metrics.sparse_categorical_accuracy])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint(
                 '../data/temp/mnist_sequential_{epoch:03d}_{val_loss:.4f}.h5',
                 save_best_only=True
             )]

history = model.fit(x_train, y_train, batch_size=128, epochs=20,
                    validation_split=0.2, callbacks=callbacks)
# Train on 48000 samples, validate on 12000 samples
# Epoch 1/20
# 48000/48000 [==============================] - 1s 29us/sample - loss: 0.4443 - accuracy: 0.8751 - val_loss: 0.2079 - val_accuracy: 0.9427
# Epoch 2/20
# 48000/48000 [==============================] - 1s 22us/sample - loss: 0.2112 - accuracy: 0.9393 - val_loss: 0.1507 - val_accuracy: 0.9584
# Epoch 3/20
# 48000/48000 [==============================] - 1s 23us/sample - loss: 0.1590 - accuracy: 0.9540 - val_loss: 0.1238 - val_accuracy: 0.9647
# Epoch 4/20
# 48000/48000 [==============================] - 1s 23us/sample - loss: 0.1302 - accuracy: 0.9616 - val_loss: 0.1083 - val_accuracy: 0.9679
# Epoch 5/20
# 48000/48000 [==============================] - 1s 22us/sample - loss: 0.1111 - accuracy: 0.9671 - val_loss: 0.0992 - val_accuracy: 0.9709
# Epoch 6/20
# 48000/48000 [==============================] - 1s 23us/sample - loss: 0.0960 - accuracy: 0.9710 - val_loss: 0.0927 - val_accuracy: 0.9719
# Epoch 7/20
# 48000/48000 [==============================] - 1s 24us/sample - loss: 0.0855 - accuracy: 0.9742 - val_loss: 0.0880 - val_accuracy: 0.9732
# Epoch 8/20
# 48000/48000 [==============================] - 1s 23us/sample - loss: 0.0748 - accuracy: 0.9772 - val_loss: 0.0809 - val_accuracy: 0.9760
# Epoch 9/20
# 48000/48000 [==============================] - 1s 24us/sample - loss: 0.0691 - accuracy: 0.9787 - val_loss: 0.0819 - val_accuracy: 0.9743
# Epoch 10/20
# 48000/48000 [==============================] - 1s 23us/sample - loss: 0.0630 - accuracy: 0.9808 - val_loss: 0.0771 - val_accuracy: 0.9758
# Epoch 11/20
# 48000/48000 [==============================] - 1s 22us/sample - loss: 0.0569 - accuracy: 0.9831 - val_loss: 0.0801 - val_accuracy: 0.9753
# Epoch 12/20
# 48000/48000 [==============================] - 1s 22us/sample - loss: 0.0518 - accuracy: 0.9844 - val_loss: 0.0778 - val_accuracy: 0.9754

print(type(history))
# <class 'tensorflow.python.keras.callbacks.History'>

print(type(history.history))
# <class 'dict'>

print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

print(history.history['accuracy'])
# [0.87514585, 0.93927085, 0.9539792, 0.96164584, 0.967125, 0.9710417, 0.9741875, 0.9771875, 0.97866666, 0.98075, 0.98310417, 0.98441666]

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(test_loss)
# 0.07156232252293267

print(test_acc)
# 0.9786

print(x_test.shape)
# (10000, 28, 28)

predictions = model.predict(x_test)

print(type(predictions))
# <class 'numpy.ndarray'>

print(predictions.shape)
# (10000, 10)

print(predictions[0])
# [3.22501933e-06 1.48081245e-08 2.89624350e-05 3.07988637e-04
#  2.93199742e-10 3.28161093e-07 2.59668814e-11 9.99620199e-01
#  5.64465245e-06 3.35659788e-05]

print(predictions[0].sum())
# 0.99999994

print(predictions[0].argmax())
# 7

results = predictions.argmax(axis=1)
print(results)
# [7 2 1 ... 4 5 6]

print(type(results))
# <class 'numpy.ndarray'>

print(results.shape)
# (10000,)

img = np.array(Image.open('../data/img/dst/mnist_sample_resize.png').resize((28, 28))) / 255
print(img.shape)
# (28, 28)

# predictions_single = model.predict(img)
# ValueError: Error when checking input: expected flatten_input to have 3 dimensions, but got array with shape (28, 28)

img_expand = img[np.newaxis, ...]
print(img_expand.shape)
# (1, 28, 28)

print(img[None, ...].shape)
# (1, 28, 28)

print(np.expand_dims(img, 0).shape)
# (1, 28, 28)

print(tf.newaxis)
# None

print(img[tf.newaxis, ...].shape)
# (1, 28, 28)

print(img[np.newaxis].shape)
# (1, 28, 28)

predictions_single = model.predict(img_expand)
print(predictions_single)
# [[6.8237398e-09 1.0004978e-08 4.0168429e-06 4.5704491e-02 2.8772252e-14
#   9.5429057e-01 5.5912805e-12 3.1738683e-08 6.9545142e-10 9.2607473e-07]]

print(predictions_single.shape)
# (1, 10)

print(predictions_single[0].argmax())
# 5

model.save('../data/temp/my_model.h5')

new_model = tf.keras.models.load_model('../data/temp/my_model.h5')
new_model.summary()
# Model: "my_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_layer (Flatten)      (None, 784)               0         
# _________________________________________________________________
# dense (Dense)                (None, 128)               100480    
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

print(new_model.evaluate(x_test, y_test, verbose=0))
# [0.07156232252293267, 0.9786]

print(new_model.predict(img_expand))
# [[6.8237398e-09 1.0004978e-08 4.0168429e-06 4.5704491e-02 2.8772252e-14
#   9.5429057e-01 5.5912805e-12 3.1738683e-08 6.9545142e-10 9.2607473e-07]]
