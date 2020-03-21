import tensorflow as tf

print(tf.__version__)
# 2.1.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, name='Layer_0', input_shape=(1000,)),
    tf.keras.layers.Dense(10, name='Layer_1'),
    tf.keras.layers.Dense(1, name='Layer_2')
], name='Sequential')

model.summary()
# Model: "Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

model.layers[1].trainable = False

model.get_layer('Layer_2').trainable = False

for l in model.layers:
    print(l.name, l.trainable)
# Layer_0 True
# Layer_1 False
# Layer_2 False

model.summary()
# Model: "Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 100,100
# Non-trainable params: 1,021
# _________________________________________________________________

for l in model.layers[1:]:
    l.trainable = True

for l in model.layers:
    print(l.name, l.trainable)
# Layer_0 True
# Layer_1 True
# Layer_2 True

model.summary()
# Model: "Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, name='Layer_0', input_shape=(1000,)),
    tf.keras.layers.Dense(10, name='Layer_1', trainable=False),
    tf.keras.layers.Dense(1, name='Layer_2')
], name='Sequential_1')

model_1.summary()
# Model: "Sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 100,111
# Non-trainable params: 1,010
# _________________________________________________________________

print(issubclass(tf.keras.Model, tf.keras.layers.Layer))
# True

model.trainable = False

for l in model.layers:
    print(l.name, l.trainable)
# Layer_0 False
# Layer_1 False
# Layer_2 False

model.summary()
# Model: "Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 0
# Non-trainable params: 101,121
# _________________________________________________________________

model.layers[1].trainable = True

for l in model.layers:
    print(l.name, l.trainable)
# Layer_0 False
# Layer_1 True
# Layer_2 False

model.summary()
# Model: "Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 0
# Non-trainable params: 101,121
# _________________________________________________________________

model.trainable = True

for l in model.layers:
    print(l.name, l.trainable)
# Layer_0 True
# Layer_1 True
# Layer_2 True

for l in model.layers:
    l.trainable = False

model.layers[1].trainable = True

for l in model.layers:
    print(l.name, l.trainable)
# Layer_0 False
# Layer_1 True
# Layer_2 False

model.summary()
# Model: "Sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_0 (Dense)              (None, 100)               100100    
# _________________________________________________________________
# Layer_1 (Dense)              (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 1,010
# Non-trainable params: 100,111
# _________________________________________________________________
