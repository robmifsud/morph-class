# File: main.py
# Author: Robert Mifsud
# Function: Main script file where all other functions are to be called from

import logging
import pandas as pd
import parameters as param
import data_cleaning as clean
import split_classes as split

# Configure logging level
# logging.basicConfig(format='%(levelname)s : %(name)s %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

# Loading Galazy Zoo 2 data and mapping from csv's into pandas DataFrames
gz2_hart = pd.read_csv(param.path_gz2_hart) # Recommended table by GZ, No Duplicates, 240k unique classifications

gz2_spec = pd.read_csv(param.path_gz2_spec) # Separate datasetes. All these objs feature in the hart table
gz2_photo = pd.read_csv(param.path_gz2_photo)
gz2_Stripe_A = pd.read_csv(param.path_gz2_82_Coadd1)
gz2_Stripe_B = pd.read_csv(param.path_gz2_82_Coadd2)
gz2_Stripe_C = pd.read_csv(param.path_gz2_82_Norm)

mapping = pd.read_csv(param.path_mapping) # Several duplicates : 355990 -> 325651
mapping = mapping[['dr7objid', 'asset_id']]

gz2_merge = clean.merge([gz2_spec, gz2_photo, gz2_Stripe_A, gz2_Stripe_B, gz2_Stripe_C])
gz2_hart = clean.merge([gz2_hart])

gz2_merge = clean.remove_duplicates(gz2_merge)
gz2_hart = clean.remove_duplicates(gz2_hart)

# print(gz2_hart.dtypes)
# print(mapping.dtypes)

# print(gz2_merge)
# print(gz2_hart)

# Join mapping with classifications
# classifications = mapping.join(gz2_hart, on='dr7objid', how='inner', lsuffix='_map')
classifications = pd.merge(mapping, gz2_hart, on='dr7objid', suffixes=("_map", None))
logging.debug(f"\n{classifications.head()}\n")
logging.debug(f"{len(classifications)}\n")
# print(classifications)

classifications.to_csv(param.path_gz2_merge, index=False)

split.get_proportions(param.gz2_3_class_regex, classifications)
split.get_proportions(param.gz2_7_class_regex, classifications)
split.get_proportions(param.gz2_9_class_regex, classifications)
split.get_proportions(param.gz2_11_class_regex, classifications)
