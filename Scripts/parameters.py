# File: parameters.py
# Author: Robert Mifsud
# Function: Used to store static values used throughout the script, for ease of use in case anything needs changing

import os

# Current path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Pointing to general 'Data' directory
dir_data = os.path.join(dir_path, '..', 'Data')
dir_temp = os.path.join(dir_data, "temp")

## Paths to datasets

# Images
path_images = os.path.join(dir_data, 'GZ2 - Images', 'images_gz2')

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
