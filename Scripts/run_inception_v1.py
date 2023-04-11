import logging, os, math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import parameters as param
import tensorflow as tf
import PIL
import PIL.Image

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

from models.inception_v1 import InceptionV1

#  **************** Config (Start) ******************

csv_logger = tf.keras.callbacks.CSVLogger(param.path_terminal_output)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

logging.info(physical_devices)

np.random.seed(123) # For reproducability

path = os.path.join(param.dir_temp, 'images_E_S_SB_224x224')

logging.info(f'✓ - Config ready')

#  **************** Config (End) ******************

#  **************** Dataset (Start) ******************

unbatched_size = 239029 # Full dataset
size = math.ceil(unbatched_size / param.BATCH_SIZE)
# size = 25000 # Smaller sample

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
full_dataset : tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels='inferred',
    # batch_size=param.BATCH_SIZE,
    batch_size=32,
    label_mode='categorical',
    # label_mode='int',
    color_mode='rgb',
    image_size=(param.IMAGE_SIZE, param.IMAGE_SIZE),
    seed=123
)


# shuffle dataset
full_dataset.shuffle(250000, seed=123, reshuffle_each_iteration=False)

# sample_dataset = full_dataset.take(size)

# train_size = int(0.8 * size)
# test_size = int(0.1 * size)
# val_size = int(0.1 * size)

# train_dataset = sample_dataset.take(train_size)
# test_dataset = sample_dataset.skip(train_size)
# val_dataset = test_dataset.skip(test_size)
# test_dataset = test_dataset.take(test_size)

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    # image = tf.image.resize(image, (227,227))
    return image, label

train_dataset = full_dataset.take(1)

train_dataset = (train_dataset.map(process_images))

class_names = full_dataset.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     # print(images[i])
#     index = tf.argmax(labels[i])
#     plt.title(class_names[index])
#     plt.axis("off")

# plt.show()

logging.info(f'✓ - Dataset loaded and split')

#  **************** Dataset (End) ******************

#  **************** Load model & config (Start) ******************

# Test one batch and see if model overfits, consider adding to separate script as well
# train_dataset = train_dataset.take(1)

# Setting callback to avoid overfitting
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)

# Callback to save weights after each epoch
path_weights = os.path.join(param.dir_weights, 'Inception_v1', 'Inception_v1.ckpt')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights,
                                                 save_weights_only=True,
                                                 verbose=1)

# Difference between load_weights and load_model, if training is halted before script is finished model will not be saved,
# ... therefore weights should be loaded to continue instead.

# Perhaps turn this into function in separate script?
model = None
if not param.LOAD_MODEL:
    logging.info(f'Compiling new {param.MODEL} model')

    model = InceptionV1()
    
    # Default learning rate 0.001
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=param.metrics)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=param.metrics)

    if param.LOAD_WEIGHTS:
        logging.info(f'Loading existing weights from {path_weights}')
        model.load_weights(path_weights)

else:
    model_path = os.path.join(param.dir_models, 'Inception_v1')
    logging.info(f'Loading {param.MODEL} from {model_path}')
    model = tf.keras.models.load_model(model_path)

logging.info(f'✓ - Model defined and loaded')

#  **************** Load model & config (End) ******************

# result = model.evaluate(test_dataset, verbose=1)

#  **************** Training (Start) ******************

# Train with validation after every epoch
# model.fit(train_dataset, epochs=param.EPOCHS, validation_data=val_dataset, verbose=1, callbacks=[earlystopping, cp_callback, csv_logger])

# Train without validation
model.fit(train_dataset, epochs=param.EPOCHS, verbose=1, callbacks=[])

logging.info(f'✓ - Training finished')

exit()

#  **************** Training (End) ******************

#  **************** Confusion Matrix (Start) ******************

# One batch for testing
res_dataset = val_dataset.take(1)

# print(f'Dataset shape: {tf.shape(res_dataset)}')

labels=[]
# tf.constant() create tensor

def extract_label(image, label):
    # logging.info(f'Imgae : {image}, Label: {label}')  
    labels.append(label)
    return image, tf.argmax(label)

res_dataset.map(extract_label)

print(f'Expected Labels: {labels}')

predict = model.predict(res_dataset, verbose=1)
print(f'Predict type: {type(predict)} Predictions: {predict}')
predict = tf.argmax(predict, axis=-1)
print(f'Predict type: {type(predict)} Predictions: {predict}')

# labels = np.concatenate([y for x, y in val_dataset], axis=0)
# con_matrix = tf.math.confusion_matrix(labels=labels, predictions=predict, num_classes=3)

#  **************** Confusion Matrix (End) ******************

# logging.info(f'Confusion Matrix (validation dataset):\n\t{con_matrix}')

# result = model.evaluate(test_dataset, verbose=1)

if param.SAVE_MODEL:
    save_location = os.path.join(param.dir_models, 'Inception_v1')

    model.save(save_location)
