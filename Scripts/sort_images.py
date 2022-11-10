import pandas as pd
import parameters as param
import images as img
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

df = pd.read_csv(param.path_gz2_merge)

img.get_images(param.gz2_3_class_regex, df)
