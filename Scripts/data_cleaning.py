# File: data_cleaning.py
# Author: Robert Mifsud
# Function: Houses functions for cleaning and merging data

import parameters as param
import pandas as pd

# Experimental only, for non gz2_hart classifications
def merge(frames : pd.DataFrame):

    # Removing features which do not concern our research
    for index, frame in enumerate(frames):
        frames[index] = frame[['dr7objid','ra','dec','rastring','decstring','sample','gz2class','total_classifications','total_votes']]

    # Merging datframes
    merge = pd.concat(frames, ignore_index=True)

    # print(merge)

    # Removing rows with no dr7objid as these cannot be mapped to an image
    merge.dropna(axis=0, how='any', subset=['dr7objid'], inplace=True)

    # Example of where to use logger.debug
    # print(merge)

    return merge

def clean_fields_3_class(df : pd.DataFrame):
    # Removing features which do not concern our research
    df = df[['dr7objid','ra','dec','rastring','decstring','sample','gz2class','total_classifications','total_votes',
             't01_smooth_or_features_a01_smooth_count', 't01_smooth_or_features_a01_smooth_fraction', 't01_smooth_or_features_a02_features_or_disk_count', 't01_smooth_or_features_a02_features_or_disk_fraction', 't01_smooth_or_features_a03_star_or_artifact_count', 't01_smooth_or_features_a03_star_or_artifact_fraction',
             't03_bar_a06_bar_count', 't03_bar_a06_bar_fraction', 't03_bar_a07_no_bar_count', 't03_bar_a07_no_bar_fraction',
             't04_spiral_a08_spiral_count', 't04_spiral_a08_spiral_fraction', 't04_spiral_a09_no_spiral_count', 't04_spiral_a09_no_spiral_fraction']]
    
    return df

# Clean data frame columns but keep agreement values
def clean_fields_ag(df : pd.DataFrame):
    df = df[['dr7objid','asset_id','gz2class','total_classifications','total_votes','agreement']]
    
    return df

def remove_duplicates(df):

    # Removing duplicates, if any
    # Example of where to place logger.info
    df.drop_duplicates(subset=['dr7objid'], inplace=True)
    
    return df
