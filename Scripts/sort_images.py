import pandas as pd
import parameters as param
import images as img
import logging, os

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

path = os.path.join(param.dir_temp, '3class_map_a(p).csv')

df = pd.read_csv(path)

# Use boolean indexing to filter rows with agreement>=0.3
df = df[df['agreement'] >= 0.3]

img.get_images(param.gz2_3_class_regex, df)
