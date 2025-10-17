import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype, is_categorical_dtype
import plotly.colors as pc
import numpy as np
import re
import math
import os
import traceback

def update_dataframe_data(df, column_name=None, category_suffix='_encoded', catNaNValue='None'):

    original_cols = list(df.columns)
    columns_to_process = df.columns if column_name is None else [column_name]
    cat_types, categorical_cols = {}, []
    for col in columns_to_process:
        unique_vals = df[col].dropna().unique()
        try:
            float_vals = [float(val) for val in unique_vals]
            #if col.lower() in plate_variables_columns: # attempt at categorizing plate variables; given up for now
            #    df[col + '_ascat'] = df[col].astype('category')
            #    cat_type = CategoricalDtype(categories=unique_vals, ordered=True)
            #    cat_types[f'cat_type_{col + '_ascat'}'] = cat_type
            #    df[col + '_ascat'] = df[col + '_ascat'].astype(cat_type)
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

    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col]) and not col.endswith(category_suffix)]
    encoded_categorical_cols = [col for col in df.columns if col.endswith(category_suffix)]
    categorical_cols = [col[:-len(category_suffix)] if col.endswith(category_suffix) else col for col in encoded_categorical_cols]

    row_var = None
    col_var = None

    if 'row'.lower() in [c.lower() for c in original_cols]:
        row_var = 'Row' if 'Row' in original_cols else 'row'
    if 'column'.lower() in [c.lower() for c in original_cols]:
        col_var = 'Column' if 'Column' in original_cols else 'column'

    return original_cols, numeric_cols, categorical_cols, row_var, col_var

def get_consistent_columns(df, subset_to_check, row_variable, column_variable):
    consistent_cols_across_rows = []
    consistent_cols_across_cols = []
    for col in df.columns:
        if col in [row_variable, column_variable] or col not in subset_to_check:
            continue
        if row_variable not in [None, 'None'] and row_variable in df.columns:
            unique_counts = df.groupby(row_variable, observed=False)[col].nunique(dropna=False)
            if (unique_counts <= 1).all():
                consistent_cols_across_rows.append(col)
        if column_variable not in [None, 'None'] and column_variable in df.columns:
            unique_counts = df.groupby(column_variable, observed=False)[col].nunique(dropna=False)
            if (unique_counts <= 1).all():
                consistent_cols_across_cols.append(col)

    return consistent_cols_across_rows, consistent_cols_across_cols

def get_discrete_colorscale(color_scale_name, num_colors):
    positions = np.linspace(0, 1, num_colors)
    return pc.sample_colorscale(color_scale_name, positions)

def lighten_rgb(rgb_str, factor): # lighten a rgb(r,g,b) string by a given factor
    m = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_str)
    if m:
        r, g, b = [int(m.group(i)) for i in range(1, 4)]
        r_new = int(r + (255 - r) * factor)
        g_new = int(g + (255 - g) * factor)
        b_new = int(b + (255 - b) * factor)
        return f'rgb({r_new},{g_new},{b_new})'
    return rgb_str

def set_graph_title(key, graph_title, split_by_variable, multi_dataframes_id_col, number_of_dfs, mdi_single_df, graph_index, dfs_delimiter):
    graph_name = None
    if dfs_delimiter in key: # we should probably correct this later as it may be in user-provided data otherwise
        series_name, split_var_name = key.split(dfs_delimiter, 1)
    else:
        series_name, split_var_name = key, None

    if graph_title:
        graph_name = graph_title.split('|', 1)[graph_index] if '|' in graph_title else graph_title
    else:
        graph_name = None

    if multi_dataframes_id_col and not mdi_single_df: # and number_of_dfs > 1:
        if graph_name:
            graph_name = graph_name + ' - ' + series_name
        else:
            graph_name = series_name
    if split_by_variable:
        if graph_name:
            graph_name = graph_name + ' - ' + split_by_variable + ' - ' + split_var_name
        else:
            graph_name = split_by_variable + ' - ' + split_var_name

    return graph_name

def set_global_map(dfs, column_name, colorscale=None, marker_symbols_patterns=None):
    global_map_output = None
    unique_values = []
    for df_i in dfs.values():
        unique_values.extend(df_i[column_name].dropna().unique())
    if colorscale:
        global_map_output = {color: get_discrete_colorscale(colorscale, len(list(dict.fromkeys(unique_values))))[i % len(list(dict.fromkeys(unique_values)))] \
                             for i, color in enumerate(list(dict.fromkeys(unique_values)))}
    elif marker_symbols_patterns:
        global_map_output = {item: marker_symbols_patterns[i % len(marker_symbols_patterns)] for i, item in enumerate(list(dict.fromkeys(unique_values)))}

    return global_map_output

def get_axis_dict(dfs, df_index, column_name, tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, category_suffix, is_y_variable=False, axis_padding=0, min_value=None, max_value=None, min_zero=True):
    axis_dict, axis_min, axis_max, axis_tickvals, axis_ticktext, axis_dtick = dict(**axis_tickstyle), None, None, None, None, None

    if column_name:
        if all(is_numeric_dtype(df[column_name]) for df in dfs.values()):
            min_value = (list(dfs.values())[df_index][column_name].min() if column_name.lower() in plate_variables_columns and df_index >=0 else min(df[column_name].min() for df in dfs.values() if column_name in df.columns)) if not min_value else min_value
            max_value = (list(dfs.values())[df_index][column_name].max() if column_name.lower() in plate_variables_columns and df_index >=0 else max(df[column_name].max() for df in dfs.values() if column_name in df.columns)) if not max_value else max_value
            axis_min = min_value if column_name.lower() in plate_variables_columns else (0 if (min_value > 0 and min_zero == True) else math.floor(min_value / tickstep) * tickstep)
            axis_max = max_value if column_name.lower() in plate_variables_columns else math.ceil(max_value/ tickstep) * tickstep
            if column_name.lower() in plate_variables_columns:
                axis_tickvals = (list(dfs.values())[df_index][column_name].dropna().unique().tolist() if df_index >= 0 else list({val for df in dfs.values() if column_name in df.columns for val in df[column_name].dropna().unique()}))
                axis_ticktext = [chr(64 + int(v)) if float(v).is_integer() else str(v) for v in axis_tickvals] if (column_name.lower() == 'row' or column_name.lower() == 'row_numeric') and plate_rows_as_alpha else [str(int(v)) for v in axis_tickvals]
                axis_dtick = 1
            else:  # ALL GLOBAL
                axis_tickvals, val = [], axis_min # this is necessary in some cases like parallel plot graphs
                while val < axis_max:
                    axis_tickvals.append(val)
                    val += tickstep
                axis_tickvals.append(axis_max)
                axis_dtick = tickstep
        elif all(is_categorical_dtype(df[column_name]) for df in dfs.values()):
            axis_min = list(dfs.values())[df_index][column_name + category_suffix].min() if df_index >= 0 else min(df[column_name + category_suffix].min() for df in dfs.values() if column_name in df.columns)
            axis_max = list(dfs.values())[df_index][column_name + category_suffix].max() if df_index >= 0 else max(df[column_name + category_suffix].max() for df in dfs.values() if column_name in df.columns)
            category_mapping = {e: c for df in dfs.values() for e, c in zip(df[column_name + category_suffix], df[column_name]) if not pd.isna(e) and not pd.isna(c)}
            axis_tickvals = list(dfs.values())[df_index][column_name + category_suffix].dropna().unique().tolist() if df_index >= 0 else list({v for df in dfs.values() if column_name in df.columns for v in df[column_name + category_suffix].dropna()})
            axis_ticktext = [category_mapping[v] for v in axis_tickvals]

            if (column_name.lower() == 'row' or column_name.lower() == 'row_numeric') and plate_rows_as_alpha and df_index >=0:
                if all(str(v).replace('.', '', 1).isdigit() for v in list(dfs.values())[df_index][column_name].dropna().unique()):
                    axis_ticktext = [chr(64 + int(float(v))) for v in list(dfs.values())[df_index][column_name].dropna().unique()]
            axis_dtick = 1

        if (column_name.lower() == 'row' or column_name.lower() == 'row_numeric') and is_y_variable:
            axis_dict['autorange'] = 'reversed'
        if axis_min is not None and axis_max is not None:
            rng = [axis_min - axis_padding, axis_max + axis_padding]
        else:
            rng = None

        axis_dict.update({
            'range':rng,
            'tickvals': axis_tickvals,
            'ticktext': axis_ticktext,
            'dtick': axis_dtick
        })

    return axis_dict

def generate_exception_message(exc: Exception) -> str:

    exception_message = 'Unknown exception.'
    tb = traceback.extract_tb(exc.__traceback__)
    user_frame = None
    for frame in reversed(tb):
        if 'site-packages' not in frame.filename:
            user_frame = frame
            break
    if user_frame:
        filename, function, line = os.path.basename(user_frame.filename), user_frame.name, user_frame.lineno
        #exception_message = f'{type(exc).__name__}: {exc} in function {function} from file {filename} at line {line}.'
        exception_message = f'Exception in function {function} from file {filename} at line {line} - {type(exc).__name__}: {str(exc).splitlines()[0]}.'

    else: # fallback to full message if nothing matches
        #exception_message = f'{type(exc).__name__}: {exc}.'
        exception_message = f'Exception - {type(exc).__name__}.'

    return exception_message

