import logging
from typing import Dict
import pandas as pd

def get_proportions(classes : Dict[str, str], df : pd.DataFrame):
    proportion_dict = {}

    logging.debug(f'Proportions for {len(classes)} classes\n')

    for gz2_class, regex in classes.items():

        # Count number of rows according to class
        proportion_dict[gz2_class] = len(df[df['gz2class'].str.match(regex)])
        percentage = (proportion_dict[gz2_class] / len(df)) * 100
        logging.debug(f'No. of {gz2_class} objects \t: {proportion_dict[gz2_class]} ({percentage:.2f}%)')

    total = 0

    for value in proportion_dict.values():
        total += value

    logging.debug('Total objects\t\t: {}\n'.format(total))

    return proportion_dict

def split_by_class(classes: Dict[str, str], df : pd.DataFrame):
    logging.info(f'Splitting dataset into {len(classes)}\n')

    df_dict = {}
    
    for gz2_class, regex in classes.items():
        df_dict[gz2_class] = df[df['gz2class'].str.match(regex)]
        logging.debug(f'Class {gz2_class} Dataframe ({len(df_dict[gz2_class])} obj.):\n {df_dict[gz2_class].head()}\n')

    return df_dict
