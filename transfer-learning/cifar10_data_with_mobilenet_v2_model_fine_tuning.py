import tensorflow as tf

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(type(x_train))
# <class 'numpy.ndarray'>

print(x_train.shape, y_train.shape)
# (50000, 32, 32, 3) (50000, 1)

print(x_test.shape, y_test.shape)
# (10000, 32, 32, 3) (10000, 1)

inputs = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (160, 160)))(inputs)
x = tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input)(x)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    weights='imagenet', input_tensor=x, input_shape=(160, 160, 3),
    include_top=False, pooling='avg'
)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# mobilenetv2_1.00_160 (Model) (None, 1280)              2257984   
# _________________________________________________________________
# dense (Dense)                (None, 10)                12810     
# =================================================================
# Total params: 2,270,794
# Trainable params: 2,236,682
# Non-trainable params: 34,112
# _________________________________________________________________

print(len(model.layers))
# 2

print(model.layers[0].name)
# mobilenetv2_1.00_160

print(len(model.layers[0].layers))
# 158

base_model.trainable = False

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# mobilenetv2_1.00_160 (Model) (None, 1280)              2257984   
# _________________________________________________________________
# dense (Dense)                (None, 10)                12810     
# =================================================================
# Total params: 2,270,794
# Trainable params: 12,810
# Non-trainable params: 2,257,984
# _________________________________________________________________

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.evaluate(x_test, y_test, verbose=0))
# [2.9224756198883055, 0.1132]

model.fit(x_train, y_train, epochs=6, validation_split=0.2, batch_size=256)
# Train on 40000 samples, validate on 10000 samples
# Epoch 1/6
# 40000/40000 [==============================] - 23s 571us/sample - loss: 1.9849 - accuracy: 0.3234 - val_loss: 1.5291 - val_accuracy: 0.4970
# Epoch 2/6
# 40000/40000 [==============================] - 21s 537us/sample - loss: 1.2436 - accuracy: 0.6140 - val_loss: 1.0953 - val_accuracy: 0.6405
# Epoch 3/6
# 40000/40000 [==============================] - 22s 540us/sample - loss: 0.9540 - accuracy: 0.6974 - val_loss: 0.9669 - val_accuracy: 0.6762
# Epoch 4/6
# 40000/40000 [==============================] - 21s 534us/sample - loss: 0.8236 - accuracy: 0.7321 - val_loss: 0.8732 - val_accuracy: 0.7070
# Epoch 5/6
# 40000/40000 [==============================] - 22s 541us/sample - loss: 0.7538 - accuracy: 0.7530 - val_loss: 0.8641 - val_accuracy: 0.7090
# Epoch 6/6
# 40000/40000 [==============================] - 22s 546us/sample - loss: 0.7110 - accuracy: 0.7629 - val_loss: 0.8390 - val_accuracy: 0.7204
# 
# <tensorflow.python.keras.callbacks.History at 0x7f79f9f37630>

print(model.evaluate(x_test, y_test, verbose=0))
# [0.8526914182662964, 0.7186]

layer_names = [l.name for l in base_model.layers]
idx = layer_names.index('block_12_expand')
print(idx)
# 110

base_model.trainable = True

for layer in base_model.layers[:idx]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# mobilenetv2_1.00_160 (Model) (None, 1280)              2257984   
# _________________________________________________________________
# dense (Dense)                (None, 10)                12810     
# =================================================================
# Total params: 2,270,794
# Trainable params: 1,812,426
# Non-trainable params: 458,368
# _________________________________________________________________

model.fit(x_train, y_train, epochs=6, validation_split=0.2, batch_size=256)
# Train on 40000 samples, validate on 10000 samples
# Epoch 1/6
# 40000/40000 [==============================] - 29s 714us/sample - loss: 0.6117 - accuracy: 0.7946 - val_loss: 0.7145 - val_accuracy: 0.7577
# Epoch 2/6
# 40000/40000 [==============================] - 26s 651us/sample - loss: 0.4992 - accuracy: 0.8292 - val_loss: 0.6788 - val_accuracy: 0.7719
# Epoch 3/6
# 40000/40000 [==============================] - 26s 656us/sample - loss: 0.4307 - accuracy: 0.8522 - val_loss: 0.6632 - val_accuracy: 0.7744
# Epoch 4/6
# 40000/40000 [==============================] - 26s 651us/sample - loss: 0.3784 - accuracy: 0.8713 - val_loss: 0.6444 - val_accuracy: 0.7792
# Epoch 5/6
# 40000/40000 [==============================] - 26s 650us/sample - loss: 0.3377 - accuracy: 0.8857 - val_loss: 0.6478 - val_accuracy: 0.7790
# Epoch 6/6
# 40000/40000 [==============================] - 27s 671us/sample - loss: 0.3038 - accuracy: 0.8981 - val_loss: 0.6257 - val_accuracy: 0.7865
# 
# <tensorflow.python.keras.callbacks.History at 0x7f79f9dbcf98>

print(model.evaluate(x_test, y_test, verbose=0))
# [0.6538689835548401, 0.7845]
