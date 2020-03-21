import tensorflow as tf

print(tf.__version__)
# 2.1.0

inner_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, name='Layer_in_0', input_shape=(1000,)),
    tf.keras.layers.Dense(10, name='Layer_in_1')
], name='Inner_model')

inner_model.summary()
# Model: "Inner_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_in_0 (Dense)           (None, 100)               100100    
# _________________________________________________________________
# Layer_in_1 (Dense)           (None, 10)                1010      
# =================================================================
# Total params: 101,110
# Trainable params: 101,110
# Non-trainable params: 0
# _________________________________________________________________

outer_model = tf.keras.Sequential([
    inner_model,
    tf.keras.layers.Dense(1, name='Layer_out_1')
], name='Outer_model')

outer_model.summary()
# Model: "Outer_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Inner_model (Sequential)     (None, 10)                101110    
# _________________________________________________________________
# Layer_out_1 (Dense)          (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

print(outer_model.layers[0] is inner_model)
# True

inner_model.layers[1].trainable = False

for l in inner_model.layers:
    print(l.name, l.trainable)
# Layer_in_0 True
# Layer_in_1 False

for l in outer_model.layers:
    print(l.name, l.trainable)
# Inner_model True
# Layer_out_1 True

outer_model.summary()
# Model: "Outer_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Inner_model (Sequential)     (None, 10)                101110    
# _________________________________________________________________
# Layer_out_1 (Dense)          (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 100,111
# Non-trainable params: 1,010
# _________________________________________________________________

outer_model.trainable = True

for l in inner_model.layers:
    print(l.name, l.trainable)
# Layer_in_0 True
# Layer_in_1 True

for l in outer_model.layers:
    print(l.name, l.trainable)
# Inner_model True
# Layer_out_1 True

outer_model.summary()
# Model: "Outer_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Inner_model (Sequential)     (None, 10)                101110    
# _________________________________________________________________
# Layer_out_1 (Dense)          (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

inner_model.trainable = False

inner_model.layers[1].trainable = True

for l in inner_model.layers:
    print(l.name, l.trainable)
# Layer_in_0 False
# Layer_in_1 True

for l in outer_model.layers:
    print(l.name, l.trainable)
# Inner_model False
# Layer_out_1 True

outer_model.summary()
# Model: "Outer_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Inner_model (Sequential)     (None, 10)                101110    
# _________________________________________________________________
# Layer_out_1 (Dense)          (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 11
# Non-trainable params: 101,110
# _________________________________________________________________

outer_model.trainable = True

functional_model = tf.keras.Model(
    inputs=inner_model.input,
    outputs=tf.keras.layers.Dense(1, name='Layer_2')(inner_model.output),
    name='Functional_model'
)

functional_model.summary()
# Model: "Functional_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_in_0_input (InputLayer [(None, 1000)]            0         
# _________________________________________________________________
# Layer_in_0 (Dense)           (None, 100)               100100    
# _________________________________________________________________
# Layer_in_1 (Dense)           (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

print(inner_model.layers[0] is functional_model.layers[1])
# True

inner_model.trainable = False

for l in functional_model.layers:
    print(l.name, l.trainable)
# Layer_in_0_input False
# Layer_in_0 False
# Layer_in_1 False
# Layer_2 True

functional_model.summary()
# Model: "Functional_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_in_0_input (InputLayer [(None, 1000)]            0         
# _________________________________________________________________
# Layer_in_0 (Dense)           (None, 100)               100100    
# _________________________________________________________________
# Layer_in_1 (Dense)           (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 11
# Non-trainable params: 101,110
# _________________________________________________________________

functional_model.layers[1].trainable = True

for l in functional_model.layers:
    print(l.name, l.trainable)
# Layer_in_0_input False
# Layer_in_0 True
# Layer_in_1 False
# Layer_2 True

functional_model.summary()
# Model: "Functional_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_in_0_input (InputLayer [(None, 1000)]            0         
# _________________________________________________________________
# Layer_in_0 (Dense)           (None, 100)               100100    
# _________________________________________________________________
# Layer_in_1 (Dense)           (None, 10)                1010      
# _________________________________________________________________
# Layer_2 (Dense)              (None, 1)                 11        
# =================================================================
# Total params: 101,121
# Trainable params: 100,111
# Non-trainable params: 1,010
# _________________________________________________________________
