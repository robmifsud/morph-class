import logging, os, math, matplotlib
import numpy as np
import parameters as param
import tensorflow as tf

from models.lenet import LeNet


csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logging.info(physical_devices)

# np.random.seed(123) # For reproducibility

path = os.path.join(param.dir_temp, 'images_E_S_SB_69x69')

unbatched_size = 239029 # Full dataset
size = math.ceil(unbatched_size / param.BATCH_SIZE)
# size = 25000 # Smaller sample

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    # image = tf.image.resize(image, (227,227))
    return image, label

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    "./mnist",
    labels='inferred',
    # batch_size=param.BATCH_SIZE,
    batch_size=2,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(69, 69),
    seed=123
)

# shuffle dataset
full_dataset.shuffle(2, seed=123, reshuffle_each_iteration=False)


# sample_dataset = full_dataset.take(size)

# train_size = int(0.8 * size)
# test_size = int(0.1 * size)
# val_size = int(0.1 * size)

# train_dataset = sample_dataset.take(train_size)
# test_dataset = sample_dataset.skip(train_size)
# val_dataset = test_dataset.skip(test_size)
# test_dataset = test_dataset.take(test_size)

# train_dataset = full_dataset.take(16)

full_dataset = (full_dataset.map(process_images))

# Test one batch and see if model overfits
# train_dataset = train_dataset.take(1)

# Setting callback to avoid overfitting
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)

# Callback to save weights after each epoch
path_weights = os.path.join(param.dir_weights, 'LeNet', 'LeNet.ckpt')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights,
                                                 save_weights_only=True,
                                                 verbose=1)

# model = LeNet((69, 69, 3), 3)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model = None
if not param.LOAD_MODEL:
    logging.info(f'Compiling new {param.MODEL} model')

    model = LeNet((69, 69, 3), 2)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    if param.LOAD_WEIGHTS:
        logging.info(f'Loading existing weights from {path_weights}')
        model.load_weights(path_weights)

else:
    model_path = os.path.join(param.dir_models, 'LeNet')
    logging.info(f'Loading {param.MODEL} from {model_path}')
    model = tf.keras.models.load_model(model_path)

# model.fit(train_dataset, epochs=param.EPOCHS, validation_data=val_dataset, verbose=1, callbacks=[earlystopping, cp_callback, csv_logger])
# Training without saving weights, logs or validation
model.fit(full_dataset, epochs=param.EPOCHS, verbose=1, callbacks=[])

# result = model.evaluate(test_dataset)

# res_dataset = val_dataset.take(1)
# labels=[]

# def extract_label(image, label):
#     # logging.info(f'Imgae : {image}, Label: {label}')  
#     labels.append(label)
#     return image, tf.argmax(label)

# res_dataset.map(extract_label)

# print(f'Expected Labels: {labels}')

# predict = model.predict(res_dataset, verbose=1)
# print(f'Predict type: {type(predict)} Predictions: {predict}')
# predict = tf.argmax(predict, axis=-1)
# print(f'Predict type: {type(predict)} Predictions: {predict}')

if param.SAVE_MODEL:
    save_location = os.path.join(param.dir_models, 'LeNet')

    model.save(save_location)
