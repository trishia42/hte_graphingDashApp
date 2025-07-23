import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
import plotly.colors as pc
import numpy as np

def convert_categorical(df, column_name=None, appended_suffix='_encoded'):
    cat_types = {}
    columns_to_process = df.columns if column_name is None else [column_name]
    for col in columns_to_process:
        unique_vals = df[col].dropna().unique()
        try:
            float_vals = [float(val) for val in unique_vals]
            continue
        except ValueError:
            pass
        cat_type = CategoricalDtype(categories=unique_vals, ordered=True)
        cat_types[f'cat_type_{col}'] = cat_type
        df[col] = df[col].astype(cat_type)
        encoded_col_name = f'{col}' + appended_suffix
        df[encoded_col_name] = df[col].cat.codes + 1
    #return df

def item_exists_in_array(columns_array, item):
    item_index = -1
    for i, val in enumerate(columns_array):  # first we check for an exact match
        if isinstance(val, str) and val.lower() == item.lower():
            item_index = i
            break
    if (item_index == -1):  # if it wasn't found, we look for the closest match and the first
        for i, val in enumerate(columns_array):  # first we check for an exact match
            if isinstance(val, str) and item.lower() in val.lower():
                item_index = i
                break
    return item_index

def check_row_column_provided(row_variable, column_variable, graph_type_string):
    if (row_variable == None) or (column_variable == None):
        print(graph_type_string, " graphs require a valid 'Row' and 'Column' columns in the dataframe/provided data file.")
        return False
    else:
        return True

def check_additional_row_column_variables(df, row_variable, column_variable, additional_row_variable, additional_column_variable):
    if additional_row_variable not in [None, 'None']:
        additional_row_variable_check = df.groupby(row_variable, observed=False)[additional_row_variable].nunique()
        if not additional_row_variable_check[additional_row_variable_check > 1].empty:
            print("Specified additional row variable is inconsistent in the dataframe and will be ignored.")
            additional_row_variable = None
    else:
        additional_row_variable = None
    if additional_column_variable not in [None, 'None']:
        additional_column_variable_check = df.groupby(column_variable, observed=False)[additional_column_variable].nunique()
        if not additional_column_variable_check[additional_column_variable_check > 1].empty:
            print("Specified additional column variable is inconsistent in the dataframe and will be ignored.")
            additional_column_variable = None
    else:
        additional_column_variable = None

    return additional_row_variable, additional_column_variable

def convert_extract_df_columns(df, column_name = None, appended_suffix='_encoded'):
    cat_types = {}
    columns_to_process = df.columns if column_name is None else [column_name]
    categorical_cols = []
    for col in columns_to_process:
        unique_vals = df[col].dropna().unique()
        try:
            float_vals = [float(val) for val in unique_vals]
            continue
        except ValueError:
            pass
        cat_type = CategoricalDtype(categories=unique_vals, ordered=True)
        cat_types[f'cat_type_{col}'] = cat_type
        df[col] = df[col].astype(cat_type)
        encoded_col_name = f'{col}' + appended_suffix
        df[encoded_col_name] = df[col].cat.codes + 1

    numeric_cols = [col for col in df.columns if
                    pd.api.types.is_numeric_dtype(df[col]) and not col.endswith(appended_suffix)]
    encoded_categorical_cols = [col for col in df.columns if col.endswith(appended_suffix)]
    categorical_cols = [col[:-len(appended_suffix)] if col.endswith(appended_suffix) else col for col in encoded_categorical_cols]

    return numeric_cols, categorical_cols, encoded_categorical_cols

def get_discrete_colorscale(color_scale_name, num_colors):
    positions = np.linspace(0, 1, num_colors)
    return pc.sample_colorscale(color_scale_name, positions)

