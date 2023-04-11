import sys
sys.path.append('../')

import parameters as param
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D , MaxPool2D ,AveragePooling2D, Dense , Dropout , Flatten , Input , GlobalAveragePooling2D, Activation, RandomRotation, RandomTranslation, RandomFlip, RandomZoom
# import tensorflow as tf
# tf.keras.layers.RandomF

def sanchez():

    model = Sequential()

    # Augmentation
    if param.AUGMENT:
        model.add(RandomRotation(0.45))
        model.add(RandomTranslation(height_factor=0.05, width_factor=0.05))
        model.add(RandomFlip(mode='horizontal_and_vertical'))
        # Not working:
        # model.add(RandomZoom(width_factor=(0.75,1.3)))

    # model.add(Conv2D(32, 6,6, border_mode='same', input_shape=(3, 69, 69)))
    model.add(Conv2D(filters=32, kernel_size=(6,6), padding='same', activation='relu', input_shape=(69, 69, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
 
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.25))

    #Neuronal Networks 
    #---------------------#

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5)) 
    # binary
    # model.add(Dense(1, init='uniform', activation='sigmoid'))

    # multi
    model.add(Dense(param.NUM_CLASSES, kernel_initializer='uniform', activation='softmax'))
   
    # print("Compilation...")
    
    # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model
