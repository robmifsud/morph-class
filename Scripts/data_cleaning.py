# File: data_cleaning.py
# Author: Robert Mifsud
# Function: Houses functions for cleaning and merging data

import parameters as param
import pandas as pd

def merge(frames : pd.DataFrame):

    # Removing features which do not concern our research
    for index, frame in enumerate(frames):
        frames[index] = frame[['dr7objid','ra','dec','rastring','decstring','sample','gz2class','total_classifications','total_votes']]

    # Merging datframes
    merge = pd.concat(frames, ignore_index=True)

    # print(merge)

    # Removing rows with no dr7objid as these cannot be mapped to an image
    merge = merge.dropna(axis=0, how='any', subset=['dr7objid'])

    # Example of where to use logger.debug
    # print(merge)

    return merge

def remove_duplicates(df):

    # Removing duplicates, if any
    # Example of where to place logger.info
    df = df.drop_duplicates(subset=['dr7objid'])
    
    return df


# # Removing features which do not concern our research
# gz2_spec_clean = gz2_spec[['dr8objid','dr7objid','ra','dec','rastring','decstring','sample','gz2class','total_classifications','total_votes']]
# gz2_photo_clean = gz2_photo[['dr8objid','dr7objid','ra','dec','rastring','decstring','sample','gz2class','total_classifications','total_votes']]

# # Merging datframes
# gz2_merge = pd.concat([gz2_spec_clean, gz2_photo_clean], ignore_index=True)
# print(gz2_merge)

# # Removing duplicates, if any. None so far(19/10/2022)
# gz2_merge = gz2_merge.drop_duplicates(subset=['dr7objid'])

# print(gz2_merge)
