# File: parameters.py
# Author: Robert Mifsud
# Function: Used to store static values used throughout the script, for ease of use in case anything needs changing

import os
from datetime import datetime
from tensorflow.keras import metrics
from tensorflow_addons.metrics import F1Score

# Runtime variables !IMPORTANT!

MODEL = 'Inception_v1'
# MODEL = 'LeNet'
RESIZE = True
OG_IMAGE_SIZE = 424
IMAGE_SIZE = 224
BATCH_SIZE = 32
LOAD_MODEL = False
LOAD_WEIGHTS = True
NUM_CLASSES = 3

# Current path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Pointing to general 'Data' directory
dir_data = os.path.join(dir_path, '..', 'Data')
dir_temp = os.path.join(dir_data, "temp")

## Paths to datasets

# Images
path_images = os.path.join(dir_data, 'GZ2 - Images', 'images_gz2', 'images')

# Mapping between images and classifications
path_mapping = os.path.join(dir_data, 'GZ2 - Images', 'gz2_filename_mapping.csv')

# GZ2 classifications
path_gz2_hart = os.path.join(dir_data,  'gz2_hart16.csv')
path_gz2_spec = os.path.join(dir_data, 'zoo2MainSpecz.csv')
path_gz2_photo = os.path.join(dir_data, 'zoo2MainPhotoz.csv')
path_gz2_82_Coadd1 = os.path.join(dir_data, 'zoo2Stripe82Coadd1.csv')
path_gz2_82_Coadd2 = os.path.join(dir_data, 'zoo2Stripe82Coadd2.csv')
path_gz2_82_Norm = os.path.join(dir_data, 'zoo2Stripe82Normal.csv')

# Path to merged dataset file
path_gz2_merge = os.path.join(dir_temp, "class_with_map.csv")

# Class prefixes
gz2_3_class         = ['E', 'S', 'SB']
gz2_3_class_regex   = {
                        'E' : '^E.*$', 
                        'S' : '^S[^B]+$', 
                        'SB': '^SB.*$'
}

gz2_7_class     = ['E', 'Sa', 'Sb', 'Sc','Se', 'SBa', 'SBc']
gz2_7_class_regex = {
                        'E'  : '^E.*$', 
                        'Sa' : '^Sa.*$',
                        'Sb' : '^Sb.*$',
                        'Sc' : '^Sc.*$',
                        'Se' : '^Se.*$',
                        'SBa': '^SBa.*$',
                        'SBc': '^SBc.*$'
}

gz2_9_class     = ['E', 'Sa', 'Sb', 'Sc','Sd', 'Se', 'SBa', 'SBc', 'SBd']
gz2_9_class_regex = {
                        'E'  : '^E.*$', 
                        'Sa' : '^Sa.*$',
                        'Sb' : '^Sb.*$',
                        'Sc' : '^Sc.*$',
                        'Sd' : '^Sd.*$',
                        'Se' : '^Se.*$',
                        'SBa': '^SBa.*$',
                        'SBc': '^SBc.*$',
                        'SBd': '^SBd.*$'
}

gz2_11_class    = ['Er','Ei','Ec', 'Sa', 'Sb', 'Sc','Sd', 'Se', 'SBa', 'SBc', 'SBd']
gz2_11_class_regex = {
                        'Er'  : '^Er.*$',  
                        'Ei'  : '^Ei.*$', 
                        'Ec'  : '^Ec.*$', 
                        'Sa' : '^Sa.*$',
                        'Sb' : '^Sb.*$',
                        'Sc' : '^Sc.*$',
                        'Sd' : '^Sd.*$',
                        'Se' : '^Se.*$',
                        'SBa': '^SBa.*$',
                        'SBc': '^SBc.*$',
                        'SBd': '^SBd.*$'
}

# Training metrics
metrics = [metrics.Accuracy(), metrics.Precision(), metrics.Recall(), metrics.MeanSquaredError(), metrics.MeanAbsoluteError(), F1Score(NUM_CLASSES)]

## Path to models
dir_models = os.path.join(dir_data, 'Models')

# Path to weights
dir_weights = os.path.join(dir_data, 'Weights')

# Path to terminal output directory
dir_terminal = os.path.join(dir_data, 'Terminal')

# Path to terminal output file
dt = datetime.now()
path_terminal_output = os.path.join(dir_terminal, f'{MODEL}_{dt.date()}_{dt.hour};{dt.minute}.csv')
