import tensorflow as tf

print(tf.__version__)
# 2.1.0

# model = tf.keras.applications.vgg16.VGG16(input_shape=(150, 150, 3))
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).

model = tf.keras.applications.vgg16.VGG16(
    include_top=False, input_shape=(150, 150, 3)
)

model.summary()
# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________

# model = tf.keras.applications.vgg16.VGG16(
#     include_top=False, input_shape=(31, 31, 3)
# )
# ValueError: Input size must be at least 32x32; got `input_shape=(31, 31, 3)`

model = tf.keras.applications.vgg16.VGG16(
    weights=None, input_shape=(150, 150, 3)
)

model.summary()
# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_2 (InputLayer)         [(None, 150, 150, 3)]     0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
# _________________________________________________________________
# flatten (Flatten)            (None, 8192)              0         
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              33558528  
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312  
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000   
# =================================================================
# Total params: 69,151,528
# Trainable params: 69,151,528
# Non-trainable params: 0
# _________________________________________________________________
