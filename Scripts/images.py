from tqdm import tqdm
from PIL import Image
import pandas as pd
import parameters as param
import os
import re
import logging
import math
import shutil

from typing import Dict
from pathlib import Path

def resize(im : Image.Image):
    # Setting the points for cropped image
    left, top, right, bottom  = 0, 0, 0, 0
    
    if param.IMAGE_SIZE == 69:
        left = 108
        top = 108
        right = 315
        bottom = 315
    else:
        x = math.floor((param.OG_IMAGE_SIZE - param.IMAGE_SIZE)/2)
        left = x
        top = x
        right = param.IMAGE_SIZE + x
        bottom = param.IMAGE_SIZE + x
    
    # Cropped image of above dimension
    # (It will not change original image)
    im = im.crop((left, top, right, bottom))

    if param.IMAGE_SIZE == 69:
        im = im.resize((69, 69), Image.Resampling.BILINEAR)
    
    # Shows the image in image viewer
    # im.show()

    return im

def get_images(classes: Dict[str, str], map : pd.DataFrame):
    folder_name = 'images'
    for key in classes.keys():
        folder_name += '_' + key

    folder_name += f'_{param.IMAGE_SIZE}x{param.IMAGE_SIZE}_a_03'
    
    path_folder = os.path.join(param.dir_temp, folder_name)

    if os.path.exists(path_folder):
        logging.info(f'Directory {path_folder} already exists')
    else:
        os.mkdir(path_folder)

    for key in classes.keys():
        path_temp = os.path.join(path_folder, key)

        if os.path.exists(path_temp):
            logging.info(f'Directory {path_temp} already exists')
        else:
            os.mkdir(path_temp)

    images = Path(param.path_images).glob('*.jpg')
    list_images = list(images)

    no_classification_count = 0

    # loop through images in directory
    for image in tqdm(list_images, total=len(list_images)):
        # Get image file name for asset id
        img_id_str = os.path.basename(image).split('.')[0]
        img_id  = int(img_id_str)

        # Open image with PIL library
        im = Image.open(os.path.join(param.dir_path, image))

        # RESIZE
        if param.RESIZE:
            im = resize(im)
        
        # Get row from mapping with matching asset_id
        obj_row = map[map['asset_id'] == img_id]
        obj_class = None

        if len(obj_row) != 0:
            # Get gz2 class for image in question
            obj_class = obj_row.iloc[0]['gz2class']
        else:
            no_classification_count += 1
            continue
        
        # Loop through classes in current classification schema
        for gz2_class, regex in classes.items():
            # if image gz2_class matches regex, put image under relevant directory
            if bool(re.search(regex, obj_class)):
                file_name = img_id_str + '.jpg'
                dst_dir = os.path.join(path_folder, gz2_class, file_name)

                # shutil.copy(image, dst_dir)
                im.save(dst_dir) 
        
    logging.debug(f'No classification count {no_classification_count}')