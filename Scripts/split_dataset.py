import sys, os, math, shutil
# sys.path.append('../')

# Importing Image class from PIL module
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import parameters as param
import numpy as np

np.random.seed(123)

# dir_ds = os.path.join(param.dir_temp, 'images_E_S_SB_69x69_a_03_train')
# dir_ds_test = os.path.join(param.dir_temp, 'images_E_S_SB_69x69_a_03_test')

# dir_ds = os.path.join(param.dir_temp, 'images_E_S_SB_227x227_a_03_train')
# dir_ds_test = os.path.join(param.dir_temp, 'images_E_S_SB_227x227_a_03_test')

dir_ds = os.path.join(param.dir_temp, 'images_E_S_SB_299x299_a_03_train')
dir_ds_test = os.path.join(param.dir_temp, 'images_E_S_SB_299x299_a_03_test')

sub_dirs = [f.path for f in os.scandir(dir_ds) if f.is_dir()]

for dir in sub_dirs:
    class_dir_name = os.path.basename(dir)
    images = Path(dir).glob('*.jpg')
    list_images = list(images)
    list_images = np.array(list_images)
    print(f'Length of {class_dir_name}: {len(list_images)}')

    size = math.floor(len(list_images)*0.1)
    list_images_test = np.random.choice(list_images, size=size, replace=False)
    print(f'Moving images in {class_dir_name} directory:')
    
    for image in tqdm(list_images_test, total=size):
        file_name = os.path.basename(image)
        dst_file = os.path.join(dir_ds_test, class_dir_name, file_name)
        shutil.move(image, dst_file)
