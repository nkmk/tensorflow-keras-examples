import tensorflow as tf

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()
# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28)]          0         
# _________________________________________________________________
# flatten (Flatten)            (None, 784)               0         
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
# Train on 48000 samples, validate on 12000 samples
# Epoch 1/20
# 48000/48000 [==============================] - 1s 26us/sample - loss: 0.4443 - accuracy: 0.8751 - val_loss: 0.2079 - val_accuracy: 0.9427
# Epoch 2/20
# 48000/48000 [==============================] - 1s 23us/sample - loss: 0.2112 - accuracy: 0.9393 - val_loss: 0.1507 - val_accuracy: 0.9584
# Epoch 3/20
# 48000/48000 [==============================] - 1s 21us/sample - loss: 0.1590 - accuracy: 0.9540 - val_loss: 0.1238 - val_accuracy: 0.9647
# Epoch 4/20
# 48000/48000 [==============================] - 2s 52us/sample - loss: 0.1302 - accuracy: 0.9616 - val_loss: 0.1083 - val_accuracy: 0.9679
# Epoch 5/20
# 48000/48000 [==============================] - 2s 47us/sample - loss: 0.1111 - accuracy: 0.9671 - val_loss: 0.0992 - val_accuracy: 0.9709
# Epoch 6/20
# 48000/48000 [==============================] - 1s 27us/sample - loss: 0.0960 - accuracy: 0.9710 - val_loss: 0.0927 - val_accuracy: 0.9719
# Epoch 7/20
# 48000/48000 [==============================] - 1s 24us/sample - loss: 0.0855 - accuracy: 0.9742 - val_loss: 0.0880 - val_accuracy: 0.9732
# Epoch 8/20
# 48000/48000 [==============================] - 2s 44us/sample - loss: 0.0748 - accuracy: 0.9772 - val_loss: 0.0809 - val_accuracy: 0.9760
# Epoch 9/20
# 48000/48000 [==============================] - 2s 36us/sample - loss: 0.0691 - accuracy: 0.9787 - val_loss: 0.0819 - val_accuracy: 0.9743
# Epoch 10/20
# 48000/48000 [==============================] - 1s 24us/sample - loss: 0.0630 - accuracy: 0.9808 - val_loss: 0.0771 - val_accuracy: 0.9758
# Epoch 11/20
# 48000/48000 [==============================] - 2s 39us/sample - loss: 0.0569 - accuracy: 0.9831 - val_loss: 0.0801 - val_accuracy: 0.9753
# Epoch 12/20
# 48000/48000 [==============================] - 1s 29us/sample - loss: 0.0518 - accuracy: 0.9844 - val_loss: 0.0778 - val_accuracy: 0.9754
# 
# <tensorflow.python.keras.callbacks.History at 0x15bbb40d0>

print(model.evaluate(x_test, y_test, verbose=0))
# [0.07156232252293267, 0.9786]
