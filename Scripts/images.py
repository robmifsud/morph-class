from tqdm import tqdm
import pandas as pd
import parameters as param
import os
import re
import logging
import shutil

from typing import Dict
from pathlib import Path

def get_images(classes: Dict[str, str], map : pd.DataFrame):
    folder_name = 'images_test'
    for key in classes.keys():
        folder_name += '_' + key

    
    path_folder = os.path.join(param.dir_temp, folder_name)

    try:
        os.mkdir(path_folder)
    except FileExistsError:
        logging.exception(f'Directory {path_folder} already exists')

    for key in classes.keys():
        path_temp = os.path.join(path_folder, key)

        try:
            os.mkdir(path_temp)
        except FileExistsError:
            logging.exception(f'Directory {path_temp} already exists')

    # images = Path(folder_dir).glob('*.png')
    images = Path(param.path_images).glob('*.jpg')
    list_images = list(images)
    # print(len(list_images))

    no_classification_count = 0

    for image in tqdm(list_images, total=len(list_images)):
        img_id  = int(os.path.basename(image).split('.')[0])
        
        obj_row = map[map['asset_id'] == img_id]
        obj_class = None

        if len(obj_row) != 0:
            obj_class = obj_row.iloc[0]['gz2class']
        else:
            no_classification_count += 1
            continue

        for gz2_class, regex in classes.items():
            if bool(re.search(regex, obj_class)):
                dst_dir = os.path.join(path_folder, gz2_class)
                shutil.copy(image, dst_dir)

        # print(obj_row.head())
        # print(obj_class)

        # print(img_id)
        
    logging.debug(f'No classification count {no_classification_count}')