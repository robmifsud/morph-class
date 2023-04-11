import logging, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import parameters as param
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(ds : tf.data.Dataset):
    all_labels = []
    for batch, labels in ds:
        labels = tf.argmax(labels, axis=1)
        all_labels.append(labels)

    all_labels = tf.concat(all_labels,0) # axis=0 as concatenating 1D tensors

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels.numpy())

    weights_dict = {index:value for index, value in enumerate(class_weights)}

    logging.info(f'✓ - Calculated class weights')

    return weights_dict

def get_class_dist(ds: tf.data.Dataset, length):
    logging.info('Getting list of target labels')
    labels=[]
    for x, y in tqdm(ds, total=length):
        labels.append(y)

    labels = tf.argmax(labels, 1)

    # Count occurences for each class
    occ = np.bincount(labels)

    logging.info(f'Bincount: {occ}')

    total = np.sum(occ)

    dist = [float(class_occ) / total for class_occ in occ]

    return dist

def get_ds_len(ds: tf.data.Dataset):
    logging.info('Getting ds length')
    labels=[]
    for x, y in tqdm(ds):
        labels.append(y)

    return len(labels)

# Visualize sample of dataset
def visualize(num_imgs, ds : tf.data.Dataset, class_names):
    # logging.info(f'Class Names: {class_names}')
    plt.figure(figsize=(10, 10))
    # ds = ds.take(1)
    for batch in ds.take(1):
        images, labels = batch
        for i in range(num_imgs):
            sq = int(math.sqrt(num_imgs))
            ax = plt.subplot(sq, sq, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            index = tf.argmax(labels[i])
            plt.title(class_names[index])
            plt.axis("off")

    plt.show(block=True)

# Normalize images to have a mean of 0 and standard deviation of 1
def normalise_images(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def confusion_matrix(model : tf.keras.Model, ds: tf.data.Dataset, class_names):
    # Predict labels with __call__ method
    y_pred = []

    # Batched
    logging.getLogger().setLevel(logging.ERROR)
    # for batch, labels in tqdm(ds,total=420):
    #     pred_batch = model(batch,training=True)
        # for x in batch:
        #     x = tf.reshape(x, shape=(1, 69, 69, 3))
        #     # logging.info(f'Type model(x) {type(model(x))}. Model(x): {model(x)}')
        #     pred_x = model(x, training=True)[0, :]
        #     y_pred.append(pred_x)
    logging.getLogger().setLevel(logging.INFO)

    # Unbatched
    # for x,y in ds:
    #     x = tf.reshape(x, shape=(1, 69, 69, 3))
    #     # logging.info(f'Type model(x) {type(model(x))}. Model(x): {model(x)}')
    #     pred_x = model(x)[0, :]
    #     y_pred.append(pred_x)
    # y_pred = np.array(y_pred)
    # y_pred = tf.argmax(y_pred, axis=1)

    # or, Predict labels with predict() method
    # steps = math.ceil(length/param.BATCH_SIZE)
    y_pred = model.predict(ds)
    y_pred = tf.argmax(y_pred, axis=1)

    # Actual labels
    y_true = tf.concat([y for x, y in ds], axis=0)
    y_true = tf.argmax(y_true, axis=1)

    # Compute the confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=y_true, predictions=y_pred, num_classes=param.NUM_CLASSES)

    # Print the confusion matrix with class names
    print('Confusion Matrix:')
    print(''.join([f'\t{class_name}' for class_name in class_names]))
    for i in range(len(class_names)):
        row = [f'{class_names[i]} (A)']
        for j in range(len(class_names)):
            row.append(str(confusion_matrix[i, j].numpy().tolist()))
        print('\t'.join(row))

# Undersampling dataset to make all class counts equal to minority class ocunts
def undersampling(ds : tf.data.Dataset):
    # This code is not efficient enough to be feasable as is
    np.empty
    x_train, y_train = [], []

    for x,y in tqdm(ds, total=param.UNBATCHED_SIZE):
        x_train.append(x)
        y_train.append(y)
    
    # x_train = np.concatenate(x_train, 0)
    # y_train = np.concatenate(y_train, 0)

    x_train, y_train = np.concatenate(list(ds.map(lambda x, y: x))), np.concatenate(list(ds.map(lambda x, y: y)))

    y_train_labels = np.argmax(y_train,axis=1)

    class_counts = np.bincount(y_train_labels)

    minority_class_count = np.min(class_counts)

    minority_class_indices = np.where(y_train_labels == np.argmin(class_counts))[0]

    # majority_class_indices = np.where(y_train_labels != np.argmin(class_counts))[0]
    ls_majority_class_indices = [np.where(y_train_labels == i)[0] for i in range(len(class_counts)) if i != np.argmin(class_counts)]

    undersample_indices = [np.random.choice(class_indices, size=minority_class_count, replace=False) for class_indices in ls_majority_class_indices]

    undersample_indices.append(minority_class_indices)
    undersample_indices = np.concatenate(undersample_indices)

    np.random.shuffle(undersample_indices)

    ds_undersampled = tf.data.Dataset.from_tensor_slices((x_train[undersample_indices], y_train[undersample_indices]))

    # ds_undersampled.shuffle(buffer_size=len(ds_undersampled), seed=123)

    return ds_undersampled

def resample(ds: tf.data.Dataset, target_dist):
    # initial_dist = get_class_dist(ds, param.UNBATCHED_SIZE)
    initial_dist =  [0.6764356916637861, 0.24890927605825913, 0.07465503227795477]

    logging.info(f'Initial dist: {initial_dist}')

    initial_tensor = tf.constant(initial_dist, dtype=tf.float32)

    target_tensor = tf.constant(target_dist, dtype=tf.float32)

    logging.getLogger().setLevel(logging.ERROR)

    ds_resample = ds.rejection_resample(class_func=lambda x,y : tf.argmax(y), initial_dist=initial_tensor, target_dist=target_tensor, seed=123)

    logging.getLogger().setLevel(logging.INFO)

    def temp_func(x,y):
        return y[0], y[1]

    ds_resample = ds_resample.map(temp_func)

    logging.info(f'✓ - Dataset resampled')

    return ds_resample

def undersample(ds: tf.data.Dataset):
    # Get the unique class labels
    unique_labels = [0,1,2]

    # Create a dictionary of datasets, with one dataset per class label
    datasets_by_class = []
    for label in unique_labels:
        # Filter the original dataset to get only examples with this label
        filtered_ds = ds.filter(lambda x,y: tf.equal(tf.argmax(y), label))
        # Add this filtered dataset to the dictionary
        datasets_by_class.append(filtered_ds)

    ds_re = tf.data.Dataset.sample_from_datasets(
        datasets_by_class,
        seed=123,
        stop_on_empty_dataset=True
    )

    return ds_re

# evaluate the model on the passed dataset
def evaluate(model: tf.keras.Model, ds: tf.data.Dataset):
    # steps = math.ceil(length/param.BATCH_SIZE)
    results = model.evaluate(ds)
    metrics = model.metrics_names
    for i in range(len(results)):
        logging.info(f'{metrics[i]}:\t{results[i]}')
