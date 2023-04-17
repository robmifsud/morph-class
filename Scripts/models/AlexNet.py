import sys
sys.path.append('../')

import parameters as param

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D , MaxPool2D ,AveragePooling2D, Dense , Dropout , Flatten , Input , GlobalAveragePooling2D, Activation, RandomRotation, RandomTranslation, RandomFlip, RandomZoom

def AlexNet():
    model = Sequential()

    # Augmentation
    if param.AUGMENT:
        model.add(RandomRotation(0.45))
        model.add(RandomTranslation(height_factor=0.05, width_factor=0.05))
        model.add(RandomFlip(mode='horizontal_and_vertical'))
        # Not working:
        # model.add(RandomZoom(width_factor=(0.75,1.3)))

    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    return model
