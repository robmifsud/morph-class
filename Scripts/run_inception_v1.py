import logging, os, math, matplotlib
import numpy as np
import parameters as param
import tensorflow as tf

from inception_v1 import InceptionV1

csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logging.info(physical_devices)

np.random.seed(123) # For reproducability

path = os.path.join(param.dir_temp, 'images_E_S_SB_224x224')

unbatched_size = 239029 # Full dataset
size = math.ceil(unbatched_size / param.BATCH_SIZE)
# size = 25000 # Smaller sample

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    batch_size=param.BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(param.IMAGE_SIZE, param.IMAGE_SIZE),
    seed=123
)

# shuffle dataset
full_dataset.shuffle(250000, seed=123, reshuffle_each_iteration=False)

sample_dataset = full_dataset.take(size)

train_size = int(0.8 * size)
test_size = int(0.1 * size)
val_size = int(0.1 * size)

train_dataset = sample_dataset.take(train_size)
test_dataset = sample_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

# Test one batch and see if model overfits, consider adding to separate script as well
# train_dataset = train_dataset.take(1)

# Setting callback to avoid overfitting
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)

# Callback to save weights after each epoch
path_weights = os.path.join(param.dir_weights, 'Inception_v1', 'Inception_v1.ckpt')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights,
                                                 save_weights_only=True,
                                                 verbose=1)

# Perhaps turn this into function in separate script?
model = None
if not param.LOAD_MODEL:
    logging.info(f'Compiling new {param.MODEL} model')

    model = InceptionV1()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=param.metrics)

    if param.LOAD_WEIGHTS:
        logging.info(f'Loading existing weights from {path_weights}')
        model.load_weights(path_weights)

else:
    model_path = os.path.join(param.dir_models, 'Inception_v1')
    logging.info(f'Loading {param.MODEL} from {model_path}')
    model = tf.keras.models.load_model(model_path)

# result = model.evaluate(test_dataset, verbose=1)

#  **************** Training (Start) ******************

# Train with validation after every epoch
model.fit(train_dataset, epochs=250, validation_data=val_dataset, verbose=1, callbacks=[earlystopping, cp_callback, csv_logger])

# Train without validation
# model.fit(train_dataset, epochs=250, verbose=1, callbacks=[earlystopping, cp_callback])

#  **************** Training (End) ******************

#  **************** Confusion Matrix (Start) ******************

# One batch for testing
# val_dataset = val_dataset.take(1)

# predict = model.predict(val_dataset, verbose=1)
# predict = tf.argmax(predict, axis=1)

# labels = np.concatenate([y for x, y in val_dataset], axis=0)
# con_matrix = tf.math.confusion_matrix(labels=labels, predictions=predict, num_classes=3)

#  **************** Confusion Matrix (End) ******************

# logging.info(f'Confusion Matrix (validation dataset):\n\t{con_matrix}')

# result = model.evaluate(test_dataset, verbose=1)

save_location = os.path.join(param.dir_models, 'Inception_v1')

model.save(save_location)
