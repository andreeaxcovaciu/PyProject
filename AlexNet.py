# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:43:16 2020

@author: covac
"""

def AlexNet():
        
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    import numpy as np
    np.random.seed(1000)
    
    
    
    #Instantiate an empty model
    model = tf.keras.Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224,224,1), kernel_size=(11,11), strides=(4,4), padding= 'valid'))
    model.add(activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid')
    model.add(activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(activation('relu'))
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid''))
    model.add(activation('relu'))
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid''))
    model.add(activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*1,)))
    model.add(activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    
    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    
    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    
    # Output Layer
    model.add(Dense(17))
    model.add(activation('softmax'))
    
    model.summary()
    
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 