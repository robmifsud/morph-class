import model as md
import os
import numpy as np
import matplotlib
import parameters as param
import tensorflow as tf

# from keras.utils import np_utils
from keras.utils import np_utils
from tensorflow.keras import callbacks
# from keras import callbacks
from model import GoogLeNet
from lenet import LeNet

# print(np.__version__)
# print(matplotlib.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

# Test is gpu is available
# devices = tf.config.list_physical_devices('GPU')
# print(f'No. of devices: {len(devices)}')

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

np.random.seed(123) # For reproducibility

path = os.path.join(param.dir_temp, 'images_E_S_SB')

# size = 239029 # Full dataset
size = 25000 # Smaller sample

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    label_mode='categorical',
    color_mode = 'rgb',
    image_size = (424, 424),
    seed = 123
    # validation_split = 0,
    # subset = 'training'
)

# shuffle dataset?
full_dataset.shuffle(250000, seed=123, reshuffle_each_iteration=False)

sample_dataset = full_dataset.take(size)

train_size = int(0.8 * size)
test_size = int(0.1 * size)
val_size = int(0.1 * size)

# train_dataset = full_dataset.take(train_size)
# test_dataset = full_dataset.skip(train_size)
# val_dataset = test_dataset.skip(test_size)
# test_dataset = test_dataset.take(test_size)

train_dataset = sample_dataset.take(train_size)
test_dataset = sample_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

train_dataset.batch(2, drop_remainder=False)
test_dataset.batch(2, drop_remainder=False)
val_dataset.batch(2, drop_remainder=False)

# Setting callback to avoid overfitting
earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

model = GoogLeNet()
# for elem in train_dataset.take(1):
    # print (f'{elem}')

# model = LeNet((424, 424, 3), 3)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=1, validation_data=val_dataset, callbacks=[earlystopping])

result = model.evaluate(test_dataset)

print(f'Test loss: {result[0]}')
print(f'Test accuracy: {result[1]}')

save_location = os.join(param.dir_models, 'first_model')

model.save(save_location)
