# File: parameters.py
# Author: Robert Mifsud
# Function: Used to store static values used throughout the script, for ease of use in case anything needs changing

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

dir_data = dir_path + '\..\Data\\'

path_images = dir_data + 'GZ2 - Images\images_gz2'
path_mapping = dir_data + 'GZ2 - Images\gz2_filename_mapping.csv'

path_gz2_hart = dir_data + 'gz2_hart16.csv'
path_gz2_spec = dir_data + 'zoo2MainSpecz.csv'
path_gz2_photo = dir_data + 'zoo2MainPhotoz.csv'
path_gz2_82_Coadd1 = dir_data + 'zoo2Stripe82Coadd1.csv'
path_gz2_82_Coadd2 = dir_data + 'zoo2Stripe82Coadd2.csv'
path_gz2_82_Norm = dir_data + 'zoo2Stripe82Normal.csv'
