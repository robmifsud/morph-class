<!-- 
    I recommend you open this file in an editor supporting markdown such as vscode which has an inbuilt preview feature (top right corner).
    https://stackedit.io/app work well too, but directory/file links will naturally not work.
 -->

# Morphological Classification of SDSS Galaxies Using Machine Learning Techniques
Author: Robert Mifsud

Supervisor: Joseph Bonello

Code used for the development of galaxy morphology classification. Developed as part of a Final Year Project in partial fulfilment of B.Sc. IT (Software Development) at the University of Malta.

## Datasets

### Images
Images in different sizes used to train classifier along with mapping to the Galaxy Zoo classifications can be found in [this Kaggle Dataset](https://www.kaggle.com/datasets/robertmifsud/resized-reduced-gz2-images) maintained by myself.
Image datasets are to be placed under [Data/temp/](./Data/temp/).

### Classifications
Classifications used to map the images to their labels datasets were sourced from the Galaxy Zoo [data releases site](https://data.galaxyzoo.org/).
The relevant dataset is under GalazyZoo2 - Table 1.

## Packages
We used the following packages during development. You may need different versions depending on your hardware. 
Our versions of TensorFlow and cuDNN had to be installed with **pip** instead of **conda**.
```
python==3.9
matplotlib
cudatoolkit=11.8.0
nvidia-cudnn-cu11==8.6.0.163
tensorflow==2.12.*
tqdm
numpy
scikit-learn
```

## Directory Structure
Scripts for running the models are under the [Scritps/](./Scripts/) directory with the ```run_``` prefix.
Other files include python scripts used to clean and sort the data.
The [Scripts/misc/](./Scripts/misc/) directory is for miscellaneous files such as matplotlib scripts.
In cases where we defined the architectures ourselves, they are kept in [Scripts/Models/](./Scripts/models/).
Scripts used on the Middle Earth Cluster are under [Scripts/CCE](./Scripts/CCE/Scripts/).

Image datasets are to be placed under [Data/temp/](./Data/temp/), so ```./Data/temp/images_E_S_SB_299x299_a_03_train/``` for example.
Model instances are saved in [Data/Models/](./Data/Models/) under the directory with their respective name and sampling method, for example ```AlexNet_under```.

CSV files with training histories are saved to [Data/Terminal/](./Data/Terminal/).

## Running the training scripts
Configuring the architecture, epochs, batch size etc. is done in the [parameters.py](./Scripts/parameters.py) file. Most possible options are left commented for the sake of convenience.
The various directories used can easily be changed in this file.
Here is an example configuration for a training run:
```
MODEL = 'inception_v3'
EPOCHS = 250
BATCH_SIZE = 32
NORMALISE = True
AUGMENT = True
OPTIMIZER = Adam()
LOAD_MODEL = False
BALANCE = 'imbalanced'
```

The [run_methods](./Scripts/run_methods.py) script includes various methods using during training such as for visualizing images, undersampling, getting metrics, confusion matrix and more.
