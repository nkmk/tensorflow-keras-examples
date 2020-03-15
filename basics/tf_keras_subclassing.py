import tensorflow as tf

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.2)
    
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

model = MyModel()

# model.summary()
# ValueError: This model has not yet been built. Build the model first by calling `build()`
# or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
# Train on 48000 samples, validate on 12000 samples
# Epoch 1/20
# 48000/48000 [==============================] - 1s 28us/sample - loss: 0.4443 - accuracy: 0.8751 - val_loss: 0.2079 - val_accuracy: 0.9427
# Epoch 2/20
# 48000/48000 [==============================] - 1s 22us/sample - loss: 0.2112 - accuracy: 0.9393 - val_loss: 0.1507 - val_accuracy: 0.9584
# Epoch 3/20
# 48000/48000 [==============================] - 1s 21us/sample - loss: 0.1590 - accuracy: 0.9540 - val_loss: 0.1238 - val_accuracy: 0.9647
# Epoch 4/20
# 48000/48000 [==============================] - 1s 21us/sample - loss: 0.1302 - accuracy: 0.9616 - val_loss: 0.1083 - val_accuracy: 0.9679
# Epoch 5/20
# 48000/48000 [==============================] - 1s 22us/sample - loss: 0.1111 - accuracy: 0.9671 - val_loss: 0.0992 - val_accuracy: 0.9709
# Epoch 6/20
# 48000/48000 [==============================] - 1s 21us/sample - loss: 0.0960 - accuracy: 0.9710 - val_loss: 0.0927 - val_accuracy: 0.9719
# Epoch 7/20
# 48000/48000 [==============================] - 1s 26us/sample - loss: 0.0855 - accuracy: 0.9742 - val_loss: 0.0880 - val_accuracy: 0.9732
# Epoch 8/20
# 48000/48000 [==============================] - 1s 21us/sample - loss: 0.0748 - accuracy: 0.9772 - val_loss: 0.0809 - val_accuracy: 0.9760
# Epoch 9/20
# 48000/48000 [==============================] - 1s 27us/sample - loss: 0.0691 - accuracy: 0.9787 - val_loss: 0.0819 - val_accuracy: 0.9743
# Epoch 10/20
# 48000/48000 [==============================] - 1s 18us/sample - loss: 0.0630 - accuracy: 0.9808 - val_loss: 0.0771 - val_accuracy: 0.9758
# Epoch 11/20
# 48000/48000 [==============================] - 1s 20us/sample - loss: 0.0569 - accuracy: 0.9831 - val_loss: 0.0801 - val_accuracy: 0.9753
# Epoch 12/20
# 48000/48000 [==============================] - 1s 24us/sample - loss: 0.0518 - accuracy: 0.9844 - val_loss: 0.0778 - val_accuracy: 0.9754
# 
# <tensorflow.python.keras.callbacks.History at 0x1582f19d0>

print(model.evaluate(x_test, y_test, verbose=0))
# [0.07156232252293267, 0.9786]

model.summary()
# Model: "my_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten (Flatten)            multiple                  0         
# _________________________________________________________________
# dense (Dense)                multiple                  100480    
# _________________________________________________________________
# dense_1 (Dense)              multiple                  1290      
# _________________________________________________________________
# dropout (Dropout)            multiple                  0         
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

model = MyModel()

model.build((None, 28, 28))

model.summary()
# Model: "my_model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_1 (Flatten)          multiple                  0         
# _________________________________________________________________
# dense_2 (Dense)              multiple                  100480    
# _________________________________________________________________
# dense_3 (Dense)              multiple                  1290      
# _________________________________________________________________
# dropout_1 (Dropout)          multiple                  0         
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _________________________________________________________________

# model.build((None, 100, 100))
# ValueError: Input 0 of layer dense_2 is incompatible with the layer: expected axis -1 of input shape to have value 784 but received input with shape [None, 10000]

model = MyModel()

model.build((None, 100, 100))

model.summary()
# Model: "my_model_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# flatten_2 (Flatten)          multiple                  0         
# _________________________________________________________________
# dense_4 (Dense)              multiple                  1280128   
# _________________________________________________________________
# dense_5 (Dense)              multiple                  1290      
# _________________________________________________________________
# dropout_2 (Dropout)          multiple                  0         
# =================================================================
# Total params: 1,281,418
# Trainable params: 1,281,418
# Non-trainable params: 0
# _________________________________________________________________
