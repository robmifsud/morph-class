import logging, os, math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Custom scripts and methods
import parameters as param
from run_methods import get_class_weights, visualize_ds, normalise_images, confusion_matrix, resample, evaluate, get_ds_len, get_class_dist, get_metrics
from models.sanchez import sanchez

csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logging.info(physical_devices)

path = os.path.join(param.dir_temp, 'images_E_S_SB_69x69_a_03_train')

size = param.UNBATCHED_SIZE

full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    shuffle=True,
    batch_size=None,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(69, 69),
    seed=123
)

class_names = full_dataset.class_names

# Normalise
if param.NORMALISE:
    full_dataset = (full_dataset.map(normalise_images))
    logging.info(f'✓ - Dataset normalized')

# Visualize dataset
# visualize_ds(full_dataset)

train_size = int(0.8888 * size)

train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

logging.info(f'Train dist: {get_class_dist(train_dataset, train_size)}')

logging.info(f'✓ - Dataset loaded and split')

if param.BALANCE == 'under':
    train_dataset = resample(ds=train_dataset, target_dist=[0.333,0.333,0.333])

train_len = get_ds_len(ds=train_dataset)
logging.info(f'Training length after any sampling: {train_len}')
train_dataset = train_dataset.repeat().batch(param.BATCH_SIZE)
steps = math.ceil(train_len/param.BATCH_SIZE)
logging.info(f'Steps: {steps}')

val_dataset = val_dataset.batch(param.BATCH_SIZE)

path_model = os.path.join(param.dir_models, f'Sanchez_{param.BALANCE}')

# Callbacks
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_model, save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1)
csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

model = None

if param.LOAD_MODEL:
    logging.info(f'Loading {param.MODEL} from {path_model}')
    logging.getLogger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(path_model)
    logging.getLogger().setLevel(logging.INFO)

else:
    logging.info(f'Compiling new {param.MODEL} model')
    model = sanchez()
    model.compile(optimizer=param.OPTIMIZER, loss='categorical_crossentropy', metrics=param.metrics)

# Training
history = model.fit(train_dataset, validation_data=val_dataset, epochs=param.EPOCHS, steps_per_epoch=steps, verbose=1, callbacks=[cp_callback, csv_logger, earlystopping])

# Testing
logging.info(f'✓ - Training finished, testing model:')

path = os.path.join(param.dir_temp, 'images_E_S_SB_69x69_a_03_test')
test_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    shuffle=False,
    batch_size=param.BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(69, 69)
)

if param.NORMALISE:
    test_dataset = test_dataset.map(normalise_images)

# Print confusion matrix
confusion_matrix(model, test_dataset, class_names)

# Report
logging.info(f'Report:\n{get_metrics(model, test_dataset, class_names)}')
