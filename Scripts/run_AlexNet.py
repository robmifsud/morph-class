import logging, os, math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import parameters as param
import tensorflow as tf

from models.AlexNet import AlexNet
from run_methods import visualize, confusion_matrix, undersampling, get_class_weights, resample, undersample, get_class_dist, get_ds_len

csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logging.info(physical_devices)

# path = os.path.join(param.dir_temp, 'images_E_S_227x227_a_03')
path = os.path.join(param.dir_temp, 'images_E_S_SB_227x227_a_03')
# path = os.path.join(param.dir_temp, 'images_E_S_227x227_check')

# unbatched_size = 239029 # Full dataset
size = 133812 # classification reduced with a(p) >= 0.3 & v >= 5

# size = math.ceil(unbatched_size / param.BATCH_SIZE)

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    return image, label

full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    shuffle=True,
    batch_size=None,
    label_mode='categorical',
    color_mode='rgb',
    image_size=(227, 227),
    seed=123
)

class_names = full_dataset.class_names

# Normalise
if param.NORMALISE:
    full_dataset = (full_dataset.map(process_images))
    logging.info(f'✓ - Dataset normalized')

train_size = int(0.8 * size)
test_size = int(0.1 * size)
val_size = int(0.1 * size)

train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

logging.info(f'✓ - Dataset loaded and split')

# train_dataset = undersampling(train_dataset)
train_dataset = resample(ds=train_dataset, target_dist=[0.333,0.333,0.333])
# train_dataset = undersample(train_dataset)

# logging.info(f'Dataset length: {get_ds_len(train_dataset)}') # 26782 (with seeds)

train_dataset = train_dataset.repeat().batch(param.BATCH_SIZE)
steps = math.ceil(26782/param.BATCH_SIZE)
logging.info(f'Steps: {steps}')
val_dataset = val_dataset.batch(param.BATCH_SIZE)

# Visualize
# visualize(16, train_dataset, class_names)

path_model = os.path.join(param.dir_models, 'AlexNet')

# Callbacks
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_model, save_weights_only=False, monitor='val_loss', save_best_only=True, verbose=1)
csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

model = None

if param.LOAD_MODEL:
    logging.info(f'Loading {param.MODEL} from {path_model}')
    model = tf.keras.models.load_model(path_model)

else:
    logging.info(f'Compiling new {param.MODEL} model')
    model = AlexNet()
    model.compile(optimizer=param.OPTIMIZER, loss='categorical_crossentropy', metrics=param.metrics)

# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_dataset, validation_data=val_dataset, epochs=param.EPOCHS, steps_per_epoch=steps, verbose=1, callbacks=[cp_callback, csv_logger, earlystopping])

# For miscellanous testing purposes (sanity checks etc.)
# train_dataset = undersampling(train_dataset)
# train_dataset.rejection_resample()

# history = model.fit(train_dataset, epochs=param.EPOCHS, verbose=1, callbacks=[])

#  Testing
logging.info(f'✓ - Training finished, testing model:')

# precision = tf.keras.metrics.Precision()
# recall = tf.keras.metrics.Recall()

# # iterate over the test dataset and update the metrics
# for x, y in test_dataset:
#     predictions = model(x)
#     precision.update_state(y, predictions)
#     recall.update_state(y, predictions)

# test_precision = precision.result().numpy()
# test_recall = recall.result().numpy()

# logging.info(f'Test precision: {test_precision}, Test recall: {test_recall}')

# Print the confusion matrix with class names
# confusion_matrix(model=model, ds=full_dataset, class_names=class_names)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2)

# Loss x Epochs
axs[0].plot(history.history['loss'])
axs[0].set_title('Training Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_ylim([0,5])

# Acc. x Epochs
axs[1].plot(history.history['accuracy'])
axs[1].set_title('Training Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')

# add some spacing between the subplots
fig.tight_layout()

plt.show()
