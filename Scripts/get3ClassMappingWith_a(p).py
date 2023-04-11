# Map Classifications from gz2_hart with images and include agreement values

import logging, re, os
from tqdm import tqdm
import pandas as pd
import parameters as param
import data_cleaning as clean
import split_classes as split
import images as im
from misc.get_agreement import agreement

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

# Load GZ2 hart classifications
gz2_hart = pd.read_csv(param.path_gz2_hart)

# Keep relevant fields
gz2_hart = clean.clean_fields_3_class(gz2_hart)

gz2_hart_with_ag = gz2_hart

count = 0 # 219494

for index, row in tqdm(gz2_hart.iterrows(), total=239695):
# for index, row in tqdm(gz2_hart.head(10000).iterrows(), total=10000):
    obj_class = row['gz2class']
    ag = 0

    if bool(re.search('^E.*$', obj_class)):
        votes = sum(row[['t01_smooth_or_features_a01_smooth_count',
                     't01_smooth_or_features_a02_features_or_disk_count',
                     't01_smooth_or_features_a03_star_or_artifact_count']].tolist())
        
        if votes < 5:
            gz2_hart_with_ag.drop(index,inplace=True)
        else:
            # Probabilities galaxy was classified as spiral, elliptical or other
            probs = row[['t01_smooth_or_features_a01_smooth_fraction',
                        't01_smooth_or_features_a02_features_or_disk_fraction',
                        't01_smooth_or_features_a03_star_or_artifact_fraction']].tolist()

            # Check for 0 values and handle as aggrement calculation involves logarithms
            c = probs.count(0)

            if c == 1: # One option is never chosen
                probs.remove(0)
                ag = agreement(probs)
            elif c == 2: # If two of three options were never chosen agreement is 1(max)
                ag = 1
            else:
                ag = agreement(probs)

            gz2_hart_with_ag.loc[index, 'agreement'] = ag

    elif bool(re.search('^S[^Be]+$', obj_class)):
        votes = sum(row[['t04_spiral_a08_spiral_count',
                     't04_spiral_a09_no_spiral_count']].tolist())

        if votes < 5:
            gz2_hart_with_ag.drop(index,inplace=True)
        else:
            probs = row[['t04_spiral_a08_spiral_fraction',
                        't04_spiral_a09_no_spiral_fraction']].tolist()
            
            c = probs.count(0)

            if c == 1:
                ag=1
            else:
                ag = agreement(probs)
        
            gz2_hart_with_ag.loc[index, 'agreement'] = ag

    elif bool(re.search('^SB.*$', obj_class)):
        votes = sum(row[['t03_bar_a06_bar_count', 
                     't03_bar_a07_no_bar_count']].tolist())
        
        if votes < 5:
            gz2_hart_with_ag.drop(index,inplace=True)
        else:
            probs = row[['t03_bar_a06_bar_fraction', 
                        't03_bar_a07_no_bar_fraction']].tolist()
            
            c = probs.count(0)

            if c == 1:
                ag=1
            else:
                ag = agreement(probs)

            gz2_hart_with_ag.loc[index, 'agreement'] = ag
    
    else:
        gz2_hart_with_ag.drop(index,inplace=True)

logging.info(f'Length of gz2_hart_with_ag: {len(gz2_hart_with_ag)}')

# Load image mappings and only keep necessary columns
mapping = pd.read_csv(param.path_mapping)
mapping = mapping[['dr7objid', 'asset_id']]

# Merge processed and reduced classifications to mapping with asset_id's
classifications = pd.merge(gz2_hart_with_ag, mapping, on='dr7objid', suffixes=("_map", None))

# Keep relevant fields
classifications = clean.clean_fields_ag(classifications)

save_path = os.path.join(param.dir_temp, '3class_map_a(p).csv')
classifications.to_csv(save_path)

# Get proportions for each class 
split.get_proportions(param.gz2_3_class_regex, classifications)
