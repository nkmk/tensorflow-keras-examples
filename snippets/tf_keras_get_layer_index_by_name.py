import tensorflow as tf
import pprint

print(tf.__version__)
# 2.1.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, name='Layer_0', input_shape=(1000,)),
    tf.keras.layers.Dense(10, name='Layer_1'),
    tf.keras.layers.Dense(1, name='Layer_2')
])

model.summary()
# Model: "sequential"
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

def get_layer_index(model, layer_name, not_found=None):
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    return not_found

print(get_layer_index(model, 'Layer_1'))
# 1

print(get_layer_index(model, 'xxxxx'))
# None

print(get_layer_index(model, 'xxxxx', -1))
# -1

print(type(model.layers))
# <class 'list'>

pprint.pprint(model.layers)
# [<tensorflow.python.keras.layers.core.Dense object at 0x12fa1a6d0>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x12fa19a90>,
#  <tensorflow.python.keras.layers.core.Dense object at 0x12fa19690>]

print(type(model.layers[0]))
# <class 'tensorflow.python.keras.layers.core.Dense'>

print(model.layers[0].name)
# Layer_0

d = {l.name: i for i, l in enumerate(model.layers)}
print(d)
# {'Layer_0': 0, 'Layer_1': 1, 'Layer_2': 2}

print(d['Layer_1'])
# 1

# print(d['xxxxx'])
# KeyError: 'xxxxx'

print(d.get('Layer_1'))
# 1

print(d.get('xxxxx'))
# None

print(d.get('xxxxx', -1))
# -1

layer_names = [l.name for l in model.layers]
print(layer_names)
# ['Layer_0', 'Layer_1', 'Layer_2']

print(layer_names.index('Layer_1'))
# 1

print([l.name for l in model.layers].index('Layer_1'))
# 1

# print([l.name for l in model.layers].index('xxxxx'))
# ValueError: 'xxxxx' is not in list

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l2 = tf.keras.layers.Dense(1, name='Layer_2')
        self.l0 = tf.keras.layers.Dense(100, name='Layer_0')
        self.l1 = tf.keras.layers.Dense(10, name='Layer_1')

    def call(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

my_model = MyModel()

my_model.build((None, 1000))

my_model.summary()
# Model: "my_model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# Layer_2 (Dense)              multiple                  11        
# _________________________________________________________________
# Layer_0 (Dense)              multiple                  100100    
# _________________________________________________________________
# Layer_1 (Dense)              multiple                  1010      
# =================================================================
# Total params: 101,121
# Trainable params: 101,121
# Non-trainable params: 0
# _________________________________________________________________

my_d = {l.name: i for i, l in enumerate(my_model.layers)}
print(my_d)
# {'Layer_2': 0, 'Layer_0': 1, 'Layer_1': 2}

print(my_d['Layer_0'])
# 1

print(my_d['Layer_1'])
# 2

print(my_d['Layer_2'])
# 0
