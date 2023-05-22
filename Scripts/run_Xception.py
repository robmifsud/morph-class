import logging, os, math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Custom scripts and methods
import parameters as param
from run_methods import get_class_weights, visualize_ds, normalise_images, confusion_matrix, resample, evaluate, get_ds_len, get_class_dist, get_metrics

csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

no_log = logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.ERROR)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logging.info(physical_devices)

path = os.path.join(param.dir_temp, 'images_E_S_SB_299x299_a_03_train')

size = param.UNBATCHED_SIZE

full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    shuffle=True,
    batch_size=None,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(299, 299),
    seed=123
)

augment_layers = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.45),
    tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    tf.keras.layers.RandomFlip(mode='horizontal_and_vertical')
])

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

# logging.info(f'Train dist: {get_class_dist(train_dataset, train_size)}')

logging.info(f'✓ - Dataset loaded and split')

if param.BALANCE == 'under':
    train_dataset = resample(ds=train_dataset, target_dist=[0.333,0.333,0.333])

len_train = get_ds_len(train_dataset)
logging.info(f'Length of train set: {len_train}')
train_dataset = train_dataset.repeat().batch(param.BATCH_SIZE)

if param.AUGMENT:
    AUTOTUNE = tf.data.AUTOTUNE
    logging.getLogger().setLevel(logging.ERROR)
    # Augment dataset with preprocessing layers
    train_dataset = train_dataset.map(lambda x, y: (augment_layers(x, training=True), y))
    # Set prefetch to preprocess and train in parallel
    train_dataset.prefetch(buffer_size=AUTOTUNE)
    logging.getLogger().setLevel(logging.INFO)

steps = math.ceil(len_train/param.BATCH_SIZE)
logging.info(f'Steps: {steps}')

val_dataset = val_dataset.batch(param.BATCH_SIZE)

path_model = os.path.join(param.dir_models, f'Xception_{param.BALANCE}')

# Callbacks
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
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
    model = tf.keras.applications.xception.Xception(classes=param.NUM_CLASSES, weights=None, classifier_activation='softmax')
    model.compile(optimizer=param.OPTIMIZER, loss='categorical_crossentropy', metrics=param.metrics)

# Training
history = model.fit(train_dataset, validation_data=val_dataset, epochs=param.EPOCHS, steps_per_epoch=steps, verbose=1, callbacks=[cp_callback, csv_logger, earlystopping])

# Testing
logging.info(f'✓ - Training finished, testing model:')

path = os.path.join(param.dir_temp, 'images_E_S_SB_299x299_a_03_test')
test_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    shuffle=False,
    batch_size=param.BATCH_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(299, 299)
)

if param.NORMALISE:
    test_dataset = test_dataset.map(normalise_images)

# Print confusion matrix
confusion_matrix(model, test_dataset, class_names)

# Report
logging.info(f'Report\n{get_metrics(model, test_dataset, class_names)}')
