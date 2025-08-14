import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
import plotly.colors as pc
import numpy as np

def item_exists_in_array(columns_array, item): #not used anywhere apparently?
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

def update_dataframe_data(df, column_name=None, category_suffix='_encoded', catNaNValue='None'):
    original_cols = list(df.columns)
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
        encoded_col_name = f'{col}' + category_suffix
        df[encoded_col_name] = df[col].cat.codes + 1
        # Handle missing categorical values
        if df[col].isna().any():
            if catNaNValue not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([catNaNValue])
            df[col] = df[col].fillna(catNaNValue)
            df[encoded_col_name] = df[col].cat.codes + 1

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and not col.endswith(category_suffix)]
    encoded_categorical_cols = [col for col in df.columns if col.endswith(category_suffix)]
    categorical_cols = [col[:-len(category_suffix)] if col.endswith(category_suffix) else col for col in encoded_categorical_cols]

    row_var = None
    col_var = None
    if (item_exists_in_array(original_cols, 'row') != -1):
        row_var = 'Row' if 'Row' in original_cols else 'row'
    if (item_exists_in_array(original_cols, 'column') != -1):
        col_var = 'Column' if 'Column' in original_cols else 'column'

    return original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_var, col_var

def get_consistent_columns(df, subset_to_check, row_variable, column_variable):
    consistent_cols_across_rows = []
    consistent_cols_across_cols = []
    for col in df.columns:
        if col in [row_variable, column_variable] or col not in subset_to_check:
            continue
        unique_counts = df.groupby(row_variable)[col].nunique(dropna=False)
        if (unique_counts <= 1).all():
            consistent_cols_across_rows.append(col)
        unique_counts = df.groupby(column_variable)[col].nunique(dropna=False)
        if (unique_counts <= 1).all():
            consistent_cols_across_cols.append(col)

    return consistent_cols_across_rows, consistent_cols_across_cols

def get_discrete_colorscale(color_scale_name, num_colors):
    positions = np.linspace(0, 1, num_colors)
    return pc.sample_colorscale(color_scale_name, positions)

