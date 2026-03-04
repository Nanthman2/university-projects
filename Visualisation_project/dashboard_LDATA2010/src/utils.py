###########################################################################
# IMPORTS
###########################################################################

import pandas as pd
import numpy as np

def format_label(var_name):
    return var_name.replace('_', ' ').title()


def detect_and_set_types(df):
    types_dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10:
                df[col] = df[col].astype('str')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                types_dict[col] = 'numerical'
        else:
            df[col] = df[col].astype('str')
    return df



def fill_categorical_missing_with_UNK(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        df[col] = df[col].fillna('UNK')
        df[col] = df[col].replace(['', 'NA', 'N/A', 'na', 'n/a', '?', 'nan', 'None'], 'UNK')
    
    return df