import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype, is_object_dtype
import plotly.graph_objects as go
from plotly.validator_cache import ValidatorCache
from plotly.subplots import make_subplots
import math
import numpy as np
import numbers
from scipy.interpolate import griddata
from miscellaneousFunctions import *

# Settings for fonts
axis_titlefont, axis_tickfont = dict(size=17, family='Times New Roman', color='darkblue', weight='bold'), dict(size=14, family='Times New Roman', color='black')
axis_tickstyle, axis_tickstep = dict(tickfont=axis_tickfont, ticks='outside', tickcolor='black', ticklen=5), 10
colorbar_titlefont, colorbar_tickfont = dict(size=15, family='Times New Roman', color='black', weight='bold'), dict(size=12, family='Times New Roman', color='black')
initial_colorbar_dict = dict(title='', tickfont=colorbar_tickfont, x=1.03, y=-0.015, xref='paper', yref='paper', xanchor='left', yanchor='bottom', lenmode='fraction', len=1.03, dtick=axis_tickstep)
graph_titlefont, hoverlabel_font =dict(color='black', family='Times New Roman', size=20, weight='bold'), dict(bgcolor='white', font_size=14, font_family='Arial', font_color='black', bordercolor='black')
legend_font = dict(family='Times New Roman', size=14)
marker_symbols = ['circle', 'square', 'diamond', 'x', 'triangle-up', 'pentagon', 'star', 'cross', 'triangle-down', 'hexagon', 'hourglass', 'triangle-se', 'triangle-sw', 'star-triangle-down', 'square-x', 'hexagram']  # preferred symbols
plotly_all_marker_symbols = marker_symbols + [symbol for symbol in ValidatorCache.get_validator('scatter.marker', 'symbol').values[2::3] if symbol not in marker_symbols]
plotly_all_marker_patterns = ['', 'x', '/', '.', '+', '|', '-', '\\']
dfs_delimiter, default_single_marker, zero_value_marker = '|=%', {'symbol': 'circle' , 'color': 'indigo', 'size':10, 'surfaceSize':2},  {'symbol': '-open', 'size':5}

def generate_parallel_coordinates_graph(dfs, parallel_variables, color_variable, colorscale, split_by_variable, plate_rows_as_alpha, multi_dataframes_id_col, number_of_dfs, graph_title, \
                                        category_suffix, plate_variables_columns):

    if color_variable:
        for key, subdf in dfs.items():
            dfs[key][color_variable] = subdf[color_variable].fillna(0).copy()  # currently we convert those rows to 0 from NaN, but do we want to drop them instead?
    dfs_combined = pd.concat(dfs.values(), ignore_index=True)

    if color_variable not in parallel_variables:
        parallel_variables.append(color_variable)

    plots=[]
    for graph_index, (key, df_i) in enumerate(dfs.items()):
        plot_dimensions = []
        for col in parallel_variables:
            if col == split_by_variable: #or col == multi_dataframes_id_col: # we don't include these
                continue
            _, axis_min, axis_max, axis_tickvals, axis_ticktext, axis_dtick = set_axis_arguments(dfs_combined, col, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle) # global to all here
            if col == color_variable:
                colorbar_min, colorbar_max = axis_min, axis_max
            range = [axis_max, axis_min] if col.lower() == 'row' else [axis_min, axis_max] # reverse if row to follow plate configuration with alpha characters
            plot_dimensions.append(dict(range=range, tickvals=axis_tickvals, ticktext=axis_ticktext, label=col, values=df_i[col + category_suffix] if isinstance(df_i[col].dtype, CategoricalDtype) else df_i[col]))

        line_dict = dict(
            color=df_i[color_variable],
            colorscale=colorscale,
            colorbar= initial_colorbar_dict.copy(),
            cmin=colorbar_min,
            cmax=colorbar_max,
        )

        fig = go.Figure(data=go.Parcoords(
            line=line_dict,
            dimensions=plot_dimensions,
            unselected=dict(line=dict(color='lightgray', opacity=0.1)),
            labelfont=axis_titlefont,
            tickfont=axis_tickfont,
        ))

        # Add colorbar title - done this way to put more space in-between the label and colorbar and have the colorbar properly aligned to the plot;
        if hasattr(fig.data[0].line, 'colorbar') and fig.data[0].line.colorbar is not None:
            if len(fig.data[0].line.colorbar.to_plotly_json()) > 0:
                fig.add_annotation(text=color_variable, font=colorbar_titlefont, textangle=0, showarrow=False, xref='paper', yref='paper', x=fig.data[0].line.colorbar.x + 0.0075, y=fig.data[0].line.colorbar.y + fig.data[0].line.colorbar.len, xanchor='left', yanchor='bottom')
        graph_title_updated = set_graph_title(key, graph_title, split_by_variable, multi_dataframes_id_col, number_of_dfs, graph_index, dfs_delimiter)
        if graph_title_updated:
            fig.add_annotation(text=graph_title_updated, font=graph_titlefont, xref='paper', yref='paper', x=-0.05, y=1.15, xanchor='left', yanchor='top', showarrow=False)
        fig.update_layout(
            margin = dict(t=110 if graph_title_updated not in [None, ''] else 60),
            shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color='darkgrey', width=1), layer='above')],
        )
        plots.append(fig)

    return plots

def generate_scatter_bubble_graph(dfs, x_variable, x_variables_subplots, y_variable, z_variable, scatter_surface, size_variable, symbol_variable, color_variable, colorscale, split_by_variable, \
                                  plate_rows_as_alpha, multi_dataframes_id_col, number_of_dfs, graph_title, category_suffix, plate_variables_columns):

    def add_hover_and_custom_data(df_or_series, custom_data, hover_lines, insert_or_append, col=None, is_numeric=False, insert_index=0, fmt=':.2f', hover_label=None, added_labels=None):
        if isinstance(df_or_series, pd.Series):
            series = df_or_series
            label = hover_label if hover_label else series.name
        elif isinstance(df_or_series, pd.DataFrame) and col:
            series = df_or_series[col]
            label = hover_label if hover_label else col
        else:
            raise ValueError('Must provide either a series or dataframe with a valid column name')

        if label in added_labels:
            return #skip addition duplicates

        if insert_or_append == 'append':
            custom_data.append(series)
            if label.lower() != 'well_label': # we want to bypass this to be consistent with other graphs; it's also unecessary
                hover_lines.append(f'{label}: %{{customdata[{len(custom_data) - 1}]{fmt if is_numeric else ''}}}')
            else:
                hover_lines.append(f'%{{customdata[{len(custom_data) - 1}]{fmt if is_numeric else ''}}}')
        else:  # insert
            custom_data.insert(insert_index, series)
            if label.lower() != 'well_label':
                hover_lines.insert(insert_index, f'{label}: %{{customdata[{insert_index}]{fmt if is_numeric else ''}}}')
            else:
                hover_lines.insert(insert_index,f'%{{customdata[{insert_index}]{fmt if is_numeric else ''}}}')

        added_labels.add(label)

    # Set some defaults
    default_size_legend_points, default_size_legend_round_value, marker_shadow, marker_shadow_offset_x, marker_shadow_offset_y, marker_shadow_add_size, axis_padding = 4, 5, False, 0.00125, 0.003125, 0.02, 0.5
    minimum_marker_px, maximum_marker_px, minimum_marker_cat_px, maximum_marker_cat_px, legend_marker_min_px, legend_marker_max_px = 5, 40, 10, 25, 5, 15 # there is a maximum in plotly legends so need to keep legend_marker_max_px small
    initial_marker_dict, subplots_max_cols = dict(line=dict(width=1, color='DarkSlateGrey'), opacity=1.0), 3
    set_axis_as_categorical = True

    scatter_subplots = bool(x_variables_subplots)
    scatter_3d = bool(z_variable and not scatter_subplots)
    if not scatter_subplots:
        # Remove any lines that are missing x/y; they will end up plotted as empty points; note that this filters/slices df in place and becomes a view (SLICE) or the original dataframe, so we would get a warning
        # when trying to perform actions on the df that you're manipulating a view; adding .copy() ensures we are then working with a new independant dataframe (which is what we want)
        for key, df_i in dfs.items():
            if x_variable not in plate_variables_columns and is_numeric_dtype(dfs[key][x_variable]): # fill values with 0
                dfs[key][x_variable] = dfs[key][x_variable].fillna(0)
            if y_variable not in plate_variables_columns and is_numeric_dtype(dfs[key][y_variable]):
                dfs[key][y_variable] = dfs[key][y_variable].fillna(0)
            dfs[key] = df_i[df_i[y_variable].notna() & df_i[x_variable].notna() & (df_i[y_variable].astype(str).str.strip() != '') & (df_i[x_variable].astype(str).str.strip() != '')].copy().reset_index(drop=True)
    else:
        marker_shadow, x_variable = False, None # for now

    dfs_combined = pd.concat(dfs.values(), ignore_index=True)

    # First we need to unify size, symbol, color across all dataframes;
    if x_variable:
        global_xaxis_dict = set_axis_arguments(dfs_combined, x_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, axis_padding=axis_padding)[0]
    global_yaxis_dict = set_axis_arguments(dfs_combined, y_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, is_y_variable=True, axis_padding=axis_padding)[0]
    if z_variable:
        global_zaxis_dict = set_axis_arguments(dfs_combined, z_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, axis_padding=axis_padding)[0]

    size_zero_mask = [None]*len(dfs)
    if size_variable:
        size_variable_column = size_variable if is_numeric_dtype(dfs_combined[size_variable]) else size_variable + category_suffix
        global_minimum_size_value, global_maximum_size_value = min(df_i[size_variable_column].min().min() for df_i in dfs.values()), max(df_i[size_variable_column].max().max() for df_i in dfs.values())
        global_size_value_range = (global_maximum_size_value - global_minimum_size_value) if (global_maximum_size_value - global_minimum_size_value) != 0 else 1
        dfs_combined['marker_size'] = (dfs_combined[size_variable_column] - global_minimum_size_value) / (global_size_value_range) * (maximum_marker_px - minimum_marker_px) + minimum_marker_px
        for i, (key, df_i) in enumerate(dfs.items()):
            if size_variable_column in df_i.columns:
                if is_numeric_dtype(dfs_combined[size_variable]):
                    dfs[key][size_variable] = dfs[key][size_variable].fillna(0)
                    size_zero_mask[i] = df_i[size_variable] == 0
                    df_i.loc[(size_zero_mask[i]), 'marker_size'] = zero_value_marker['size']
                else:
                    size_zero_mask[i] = pd.Series(False, index=df_i.index)
                df_i['marker_size'] = (df_i[size_variable_column] - global_minimum_size_value) / (global_size_value_range) * (maximum_marker_px - minimum_marker_px) + minimum_marker_px
    else:
        for i, (key, df_i) in enumerate(dfs.items()):
            df_i['marker_size'] = default_single_marker['surfaceSize'] if scatter_3d and scatter_surface else default_single_marker['size']
            size_zero_mask[i] = pd.Series(False, index=df_i.index)

    symbol_zero_mask = [None]*len(dfs)
    if symbol_variable:
        global_marker_symbol_map = {item: plotly_all_marker_symbols[i % len(plotly_all_marker_symbols)] for i, item in enumerate(dfs_combined[symbol_variable].unique())}
        for i, (key, df_i) in enumerate(dfs.items()):
            if symbol_variable in df_i.columns:
                if is_numeric_dtype(dfs_combined[symbol_variable]):
                    dfs[key][symbol_variable] = dfs[key][symbol_variable].fillna(0)
                    symbol_zero_mask[i] = df_i[symbol_variable] == 0
                else:
                    symbol_zero_mask[i] = pd.Series(False, index=df_i.index)
                df_i['marker_symbol'] = df_i[symbol_variable].map(global_marker_symbol_map)
    else:
        for i, (key, df_i) in enumerate(dfs.items()):
            df_i['marker_symbol'] = default_single_marker['symbol']
            symbol_zero_mask[i] = pd.Series(False, index=df_i.index)

    for i, (key, df_i) in enumerate(dfs.items()):
        zero_mask = size_zero_mask[i] | symbol_zero_mask[i]
        if isinstance(df_i['marker_symbol'], pd.Series):
            df_i['marker_symbol'] = df_i['marker_symbol'].astype(str)
            df_i.loc[zero_mask, 'marker_symbol'] = (df_i.loc[zero_mask, 'marker_symbol'] + zero_value_marker['symbol']) if '-open' not in df_i.loc[zero_mask, 'marker_symbol'] else df_i.loc[zero_mask, 'marker_symbol']
        else:
            marker_symbols_array = np.full(len(df_i), df_i['marker_symbol'], dtype=object)
            marker_symbols_array[zero_mask] = [str(sym) + zero_value_marker['symbol'] if '-open' not in str(sym) else str(sym) for sym in marker_symbols_array[zero_mask]]

    global_marker_color_map = None
    if color_variable:
        if not is_numeric_dtype(dfs_combined[color_variable]):
            global_marker_color_map = {color: get_discrete_colorscale(colorscale, len(dfs_combined[color_variable].dropna().unique()))[i % len(get_discrete_colorscale(colorscale, len(dfs_combined[color_variable].dropna().unique())))]
                for i, color in enumerate(dfs_combined[color_variable].dropna().unique())}
            for key, df_i in dfs.items():
                if color_variable in df_i.columns:
                    df_i['marker_color'] = df_i[color_variable].map(global_marker_color_map)
        else:
            _, global_colorbar_cmin, global_colorbar_cmax, _, _, _ = set_axis_arguments(dfs_combined, color_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle)
            for key, df_i in dfs.items():
                if color_variable in df_i.columns:
                    dfs[key][color_variable] = dfs[key][color_variable].fillna(0)
                    df_i['marker_color'] = df_i[color_variable]
    else:
        for key, df_i in dfs.items():
            df_i['marker_color'] = default_single_marker['color']

    plots = []
    max_figure_columns, max_figure_rows = 0, 0
    for graph_index, (key, df_i) in enumerate(dfs.items()):
        if x_variable:
            local_xaxis_dict = set_axis_arguments(df_i, x_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, axis_padding=axis_padding)[0] if \
                x_variable.lower() in plate_variables_columns else global_xaxis_dict
        local_yaxis_dict = set_axis_arguments(df_i, y_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, is_y_variable=True, axis_padding=axis_padding)[0] if \
            y_variable.lower() in plate_variables_columns else global_yaxis_dict
        if z_variable:
            local_zaxis_dict = set_axis_arguments(df_i, z_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, axis_padding=axis_padding)[0] if \
                z_variable.lower() in plate_variables_columns else global_zaxis_dict

        custom_data, hover_lines, added_labels = [], [], set()
        marker_dict = initial_marker_dict.copy()
        if df_i.empty:
            continue
        if not scatter_subplots: # may need to combine this in the future and also allow more flexibility with alpha/row/column assignments
            if y_variable.lower() == 'row' and plate_rows_as_alpha:  # convert rows to alphanumeric characters for the hover template
                if x_variable and x_variable.lower() == 'column':
                    # df.apply applies a function across the dataframe, lambda creates an unnamed function (here taking a single argument 'row' and axis=1 means it's applied row by row, axis=0 would be column by column
                    df_i['well_label'] = df_i.apply(lambda row: f'{chr(64 + int(row[y_variable]))}{int(row[x_variable])}' if plate_rows_as_alpha else f'{int(row[y_variable])}, {int(row[x_variable])}', axis=1)
                    add_hover_and_custom_data(df_i, custom_data, hover_lines, 'insert', 'well_label', False, 0, None, added_labels=added_labels)
                else:
                    # here we select a single column in the dataframe and it's applied element-by-element (one-dimensional series)
                    df_i['row_letter'] = df_i[y_variable].apply(lambda r: chr(64 + int(r))) if plate_rows_as_alpha else df_i[y_variable]
                    add_hover_and_custom_data(df_i, custom_data, hover_lines, 'insert', 'row_letter', False, 0, None, y_variable, added_labels)
                    hover_lines.insert(1, f'{x_variable}: %{{x}}')
            else:  # just put x, y
                hover_lines.extend([f'{y_variable}: %{{y}}', f'{x_variable}: %{{x}}'])
                added_labels.add(x_variable)
                if (y_variable.lower() != x_variable.lower()):
                    added_labels.add(y_variable)
        else:
            hover_lines.extend([f'{y_variable}: %{{y}}']) # just add y here
            added_labels.add(y_variable)

        if x_variable:
             x_vals = df_i[x_variable] if not isinstance(df_i[x_variable].dtype, CategoricalDtype) else df_i[x_variable + category_suffix]
        y_vals = df_i[y_variable] if not isinstance(df_i[y_variable].dtype, CategoricalDtype) else df_i[y_variable + category_suffix]

        if x_variable:
            if (isinstance(df_i[x_variable].dtype, CategoricalDtype) or x_variable.lower() in plate_variables_columns):
                if df_i[x_variable].nunique(dropna=True) > max_figure_columns:
                    max_figure_columns = df_i[x_variable].nunique(dropna=True)
        if (isinstance(df_i[y_variable].dtype, CategoricalDtype) or y_variable.lower() in plate_variables_columns):
            if df_i[y_variable].nunique(dropna=True) > max_figure_rows:
                max_figure_rows= df_i[y_variable].nunique(dropna=True)

        if scatter_3d:
            z_vals = df_i[z_variable + category_suffix] if (isinstance(df_i[z_variable].dtype, CategoricalDtype)) else df_i[z_variable]
            hover_lines.extend([f'{z_variable}: %{{z}}'])
            if (z_variable.lower() != x_variable.lower() and z_variable.lower() != y_variable.lower()):
                added_labels.add(z_variable)

        if size_variable not in [None, 'None']:
            add_hover_and_custom_data(df_i, custom_data, hover_lines, 'append', size_variable, is_numeric_dtype(df_i[size_variable]), added_labels=added_labels)
        if color_variable not in [None, 'None']:
            add_hover_and_custom_data(df_i, custom_data, hover_lines, 'append', color_variable, is_numeric_dtype(df_i[color_variable]), added_labels=added_labels)
        if symbol_variable not in [None, 'None']:
            if is_numeric_dtype(df_i[symbol_variable]):
                add_hover_and_custom_data(df_i, custom_data, hover_lines, 'append', symbol_variable, True, added_labels=added_labels)
            else:
                hover_lines.append(f'{symbol_variable}: %{{text}}')
            text_data = df_i[symbol_variable]
        else:
            text_data = df_i[x_variable] if x_variable else None

        if scatter_subplots:
            if len(x_variables_subplots) <= subplots_max_cols:
                subplots_rows, subplots_columns = 1, len(x_variables_subplots)
            else:
                subplots_rows, subplots_columns = (len(x_variables_subplots) // subplots_columns) + 1, subplots_max_cols
            fig = make_subplots(rows=subplots_rows, cols=subplots_columns)
        else:
            fig = go.Figure()

        marker_dict.update(dict(size=df_i['marker_size'], symbol=df_i['marker_symbol']))
        if color_variable:
            if is_numeric_dtype(df_i[color_variable]):
                marker_dict.update(dict(color=df_i[color_variable], cmin=global_colorbar_cmin , cmax=global_colorbar_cmax, showscale=True, colorscale=colorscale, colorbar=initial_colorbar_dict.copy()))
            else:
                marker_dict.update(dict(color=df_i['marker_color']))
        else:
            marker_dict.update(dict(color=df_i['marker_color']))

        if marker_shadow:  # Visual effect for fun - was disabled up top since breaks if we make the numerical axis categorical...
            marker_dict_shadow = marker_dict.copy()
            marker_dict_shadow.update(size=marker_dict['size'] + (marker_dict['size']*marker_shadow_add_size), color='rgba(50, 50, 50, 0.35)', opacity=1.0, line=dict(width=0), showscale=False)
            if (isinstance(df_i[x_variable].dtype, CategoricalDtype)):
                marker_shadow_offset_x_dfi = marker_shadow_offset_x * len(df_i[x_variable].cat.categories)
                x_vals_shadow = [x_val + marker_shadow_offset_x_dfi for x_val in df_i[x_variable + category_suffix]]
            else:
                marker_shadow_offset_x_dfi = marker_shadow_offset_x * df_i[x_variable].nunique()
                x_vals_shadow = [x_val + marker_shadow_offset_x_dfi for x_val in df_i[x_variable]]
            if (isinstance(df_i[y_variable].dtype, CategoricalDtype)):
                marker_shadow_offset_y_dfi = marker_shadow_offset_y * len(df_i[y_variable].cat.categories)
                y_vals_shadow = ([y_val + marker_shadow_offset_y_dfi for y_val in df_i[y_variable + category_suffix]] if (y_variable.lower() == 'row') else [y_val - marker_shadow_offset_y_dfi / 2 for y_val in df_i[y_variable + category_suffix]])
            else:
                marker_shadow_offset_y_dfi = marker_shadow_offset_y * df_i[y_variable].nunique()
                y_vals_shadow = ([y_val + marker_shadow_offset_y_dfi for y_val in df_i[y_variable]] if (y_variable.lower() == 'row') else [y_val - marker_shadow_offset_y_dfi / 2 for y_val in df_i[y_variable]])
            if scatter_3d:
                fig.add_trace(go.Scatter3d(x=x_vals_shadow, y=y_vals_shadow, z=z_vals, mode='markers', marker=marker_dict_shadow, name='', showlegend=False, hoverinfo='skip'))
            else:
                fig.add_trace(go.Scatter(x=x_vals_shadow, y=y_vals_shadow, mode='markers', marker=marker_dict_shadow, name='', showlegend=False, hoverinfo='skip'))

        # combines list of arrays into a single 2D numpy array; axis=-1 means stacking is done column-wise to each row
        if scatter_3d:
            if scatter_surface:
                x_vals_data = df_i[x_variable + category_suffix] if isinstance(df_i[x_variable].dtype, CategoricalDtype) else df_i[x_variable]
                y_vals_data = df_i[y_variable + category_suffix] if isinstance(df_i[y_variable].dtype, CategoricalDtype) else df_i[y_variable]
                z_vals_data = df_i[z_variable + category_suffix] if isinstance(df_i[z_variable].dtype, CategoricalDtype) else df_i[z_variable]
                x_vals_surface, y_vals_surface = np.meshgrid(np.linspace(x_vals_data.min(), x_vals_data.max(), 100), np.linspace(y_vals_data.min(), y_vals_data.max(), 100))
                z_vals_surface = griddata((x_vals_data, y_vals_data), z_vals_data, (x_vals_surface, y_vals_surface), method='linear')  # Interpolate z values onto grid
                surface_color = None
                if color_variable:
                    if is_numeric_dtype(dfs_combined[color_variable]):
                        surface_colorscale = colorscale
                    else:
                        surface_colorscale = []
                        surface_colorscale = [[(i - 1) / (len(dfs_combined[color_variable].cat.categories) - 1), global_marker_color_map[cat]] \
                                                 for i, cat in enumerate(dfs_combined[color_variable].cat.categories, start=1)]
                        surface_color = griddata((x_vals_data, y_vals_data), df_i[color_variable + category_suffix], (x_vals_surface, y_vals_surface), method='linear')
                else:
                    surface_colorscale = [[0, default_single_marker['color']], [1, default_single_marker['color']]]
                fig.add_trace(go.Surface(x=x_vals_surface, y=y_vals_surface, z=z_vals_surface, opacity=0.5, surfacecolor=surface_color, colorscale=surface_colorscale, showscale=False))
            fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='markers', marker=marker_dict, text=text_data, customdata=np.stack(custom_data, axis=-1) if custom_data else None, \
                                       hovertemplate='<br>'.join(hover_lines) + '<extra></extra>', name='', showlegend=False))
        else:
            if scatter_subplots:
                subplot_row, subplot_col = 1, 1
                for k, x_var in enumerate(x_variables_subplots):
                    x_subplot_data = df_i[x_var].cat.codes + 1 if isinstance(df_i[x_var].dtype, pd.CategoricalDtype) else df_i[x_var]
                    fig.add_trace(go.Scatter(x=x_subplot_data, y=df_i[y_variable], mode='markers', marker=marker_dict, text=text_data, customdata=np.stack(custom_data, axis=-1) if custom_data else None, hovertemplate='<br>'.join(hover_lines) + '<extra></extra>', \
                                             name='', showlegend=False),row=subplot_row, col=subplot_col)
                    # we use local no matter what for x_axis and update it for the x_var loop
                    local_xaxis_dict = set_axis_arguments(df_i, x_var, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, axis_padding=axis_padding)[0]
                    fig.update_xaxes(title_text=x_var, **local_xaxis_dict, row=subplot_row, col=subplot_col)
                    fig.update_yaxes(title_text=y_variable, **local_yaxis_dict, row=subplot_row, col=subplot_col)
                    subplot_row, subplot_col = subplot_row if subplot_col <= subplots_max_cols else subplot_row + 1, subplot_col + 1 if subplot_col < subplots_max_cols else 1
                for k in range(1, len(x_variables_subplots) + 1):
                    x_domain = fig.layout[f"xaxis{k if k > 1 else ''}"].domain
                    y_domain = fig.layout[f"yaxis{k if k > 1 else ''}"].domain
                    fig.add_shape(type='rect', xref='paper', yref='paper', x0=x_domain[0], x1=x_domain[1], y0=y_domain[0], y1=y_domain[1], line=dict(color='black', width=2), fillcolor='rgba(0,0,0,0)')
            else:
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', marker=marker_dict, text=text_data, customdata=np.stack(custom_data, axis=-1) if custom_data else None, \
                                        hovertemplate='<br>'.join(hover_lines) + '<extra></extra>', name='', showlegend=False))
        # Final layout adjustments
        if not scatter_3d and color_variable:
            colorbar_trace_index = 0 + int(marker_shadow) if fig.data[0 + int(marker_shadow)].marker.showscale else None
        elif scatter_3d and color_variable: # and not scatter_surface:
            colorbar_trace_index = 0 + int(marker_shadow) + int(scatter_surface) if fig.data[0 + int(marker_shadow) + int(scatter_surface)].marker.showscale else None
        else:
            colorbar_trace_index = None
        if colorbar_trace_index:
            fig.add_annotation(text=color_variable, font=colorbar_titlefont, textangle=0, showarrow=False, xref='paper', yref='paper', \
                                x=(fig.data[colorbar_trace_index].marker.colorbar.x + 0.0075), \
                                y=(fig.data[colorbar_trace_index].marker.colorbar.y + fig.data[colorbar_trace_index].marker.colorbar.len),  \
                                xanchor='left', yanchor='bottom')

        graph_title_updated = set_graph_title(key, graph_title, split_by_variable, multi_dataframes_id_col, number_of_dfs, graph_index, dfs_delimiter)
        if graph_title_updated:
            fig.add_annotation(text=graph_title_updated, font=graph_titlefont, xref='paper', yref='paper', x=-0.075, y=1.1, xanchor='left', yanchor='top', showarrow=False)
        top_margin = 70 if graph_title_updated not in [None, ''] else 60
        fig.update_layout(
            margin=dict(t=top_margin, l=10, r=10, b=10),
            legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='right', x=1, font=legend_font, itemsizing='trace'),
            template='plotly_white',
            hoverlabel=hoverlabel_font,
            autosize=True
        )

        if not scatter_subplots: # otherwise already done in loop;
            if not scatter_3d:
                fig.update_layout(
                    xaxis=dict(title=dict(text=x_variable, font=axis_titlefont, standoff=30), **local_xaxis_dict),
                    yaxis=dict(title=dict(text=y_variable, font=axis_titlefont, standoff=30), **local_yaxis_dict),
                    shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1), layer='above')],
                )
                if set_axis_as_categorical:
                    if x_variable.lower() in plate_variables_columns:
                        fig.update_xaxes(range=None, type='category', categoryorder='array', categoryarray=df_i[x_variable])
                    if y_variable.lower() in plate_variables_columns:
                        fig.update_yaxes(range=None, type='category', categoryorder='array', categoryarray=df_i[y_variable])
            else: # scatter_3d
                fig.update_layout(
                    scene=dict(bgcolor='white',
                            xaxis = dict(title=dict(text=x_variable, font=axis_titlefont), **local_xaxis_dict, showbackground=True, backgroundcolor='lightgrey', gridcolor='white', gridwidth=2, zerolinecolor='black', zerolinewidth=3, showspikes=False),
                            yaxis = dict(title=dict(text=y_variable, font=axis_titlefont), **local_yaxis_dict, showbackground=True, backgroundcolor='lightgrey', gridcolor='white', gridwidth=2, zerolinecolor='black', zerolinewidth=3),
                            zaxis = dict(title=dict(text=z_variable, font=axis_titlefont), **local_zaxis_dict, showbackground=True, backgroundcolor='lightgrey', gridcolor='white', gridwidth=2, zerolinecolor='black', zerolinewidth=3)
                            ),
                    scene_camera=dict(eye=dict(x=2, y=-2, z=0.8)),
                )

        # Add dummy traces for symbol, size, color variables so that we can have legends for them
        if symbol_variable and symbol_variable != split_by_variable:
            for item, symbol in global_marker_symbol_map.items():
                display_name = f'{item:.1f}' if isinstance(item, (int, float, numbers.Integral)) else str(item)
                if not scatter_3d:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol=symbol, size=10, color='black'), name=display_name, legendgroup='symbol_legend', showlegend=True))
                else:
                    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(symbol=symbol, size=10, color='black'), name=display_name, legendgroup='symbol_legend', showlegend=True, hoverinfo='skip'))

        size_legend_values = None
        if size_variable and size_variable != split_by_variable:
            if is_numeric_dtype(dfs_combined[size_variable]):
                legend_min = max(default_size_legend_round_value, math.floor(global_minimum_size_value / default_size_legend_round_value) * default_size_legend_round_value)
                legend_max = math.ceil(global_maximum_size_value / default_size_legend_round_value) * default_size_legend_round_value
                vals = np.linspace(legend_min, legend_max, default_size_legend_points)
                vals = [round(val / default_size_legend_round_value) * default_size_legend_round_value for val in vals]
                size_legend_values = [(val, ((val - legend_min) / (legend_max - legend_min) if legend_max > legend_min else 0) * (legend_marker_max_px - legend_marker_min_px) + legend_marker_min_px)
                    for val in vals
                ]
            else:
                category_sizes = dfs_combined[[size_variable, 'marker_size']].drop_duplicates().sort_values('marker_size')
                category_sizes['legend_marker_size_px'] = ((category_sizes['marker_size'] - category_sizes['marker_size'].min()) / (category_sizes['marker_size'].max() - category_sizes['marker_size'].min()) * (legend_marker_max_px - legend_marker_min_px) + legend_marker_min_px
                    if category_sizes['marker_size'].max() > category_sizes['marker_size'].min() else legend_marker_min_px)
                size_legend_values = list(zip(category_sizes[size_variable], category_sizes['legend_marker_size_px']))  # creates a list of tuples from the two columns in cat_sizes
            for val, px in size_legend_values:
                val_str = f'{val:.1f}' if isinstance(val, (int, float, numbers.Integral)) else str(val)
                display_name = f'{size_variable} = {val_str}' if size_variable else val_str
                if not scatter_3d:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=px, color='black', opacity=0.5, line=dict(width=0.5, color='gray')), name=display_name, \
                                             legendgroup='size_legend', showlegend=True))
                else:
                    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=px, color='black', opacity=0.5, line=dict(width=0.5, color='gray')), name=display_name, \
                                               legendgroup='size_legend', showlegend=True))

        if global_marker_color_map and color_variable != split_by_variable:
            for item, color in global_marker_color_map.items():
                if not scatter_3d:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=str(item), legendgroup='color_legend', showlegend=True))
                else:
                    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=10, color=color), name=str(item), legendgroup='color_legend', showlegend=True))

        plots.append(fig)

    return plots

def generate_heatmap_graph(dfs, row_variable, column_variable, color_variable, additional_row_variable, additional_column_variable, z_smooth_option, colorscale, split_by_variable, plate_rows_as_alpha, \
                           multi_dataframes_id_col, number_of_dfs, graph_title, plate_variables_columns):

    # Remove any lines that are missing row/column, clean up additional_row/column_variables;
    for key, df_i in dfs.items():
        dfs[key] = df_i[df_i[row_variable].notna() & df_i[column_variable].notna() & (df_i[row_variable].astype(str).str.strip() != '') & (df_i[column_variable].astype(str).str.strip() != '')].copy()
    dfs_combined = pd.concat(dfs.values(), ignore_index=True)
    dfs_combined = dfs_combined[dfs_combined[row_variable].notna() & dfs_combined[column_variable].notna() & (dfs_combined[row_variable].astype(str).str.strip() != '') & (dfs_combined[column_variable].astype(str).str.strip() != '')].copy()

    additional_row_variable = additional_row_variable if isinstance(additional_row_variable, list) else ([] if additional_row_variable else [additional_row_variable])
    additional_column_variable = additional_column_variable if isinstance(additional_column_variable, list) else ([] if additional_column_variable else [additional_column_variable])

    plots = []
    max_figure_columns, max_figure_rows = 0, 0
    x_domain, y_domain = [0, 0.90], [0, 1] # attempt at sizing graphs proportionally...
    for graph_index, (key, df_i) in enumerate(dfs.items()):
        # disabled above for now; will split dataframes if they exist, but left the code as a placeholder for now
        #add_new_figure = False
        ##if multi_dataframes_id_col:
        ##    add_new_figure = True
        #if not split_by_variable:
        #    add_new_figure = (i == 0)
        #else:
        #    add_new_figure = (i < len(dfs) / number_of_dfs)
        add_new_figure = True

        z_hover_labels = df_i.pivot(index=row_variable, columns=column_variable, values=color_variable)
        _, global_zaxis_min, global_zaxis_max, global_zaxis_tickvals, global_zaxis_ticktext, _ = set_axis_arguments(dfs_combined, color_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle)
        all_rows, all_cols = df_i[row_variable].dropna().unique(), df_i[column_variable].dropna().unique()
        if (isinstance(df_i[color_variable].dtype, CategoricalDtype)):
            z_categorical = df_i.pivot_table(index=row_variable, columns=column_variable, values=color_variable, aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            category_to_number = {cat: i + 1 for i, cat in enumerate(list(df_i[color_variable].cat.categories))}
            heatmap_data = z_categorical.map(lambda val: category_to_number.get(val, np.nan))
            colorscale = get_discrete_colorscale(colorscale, len(list(dfs_combined[color_variable].cat.categories)))
            show_colorbar = False
        else:
            heatmap_data = df_i.pivot_table(index=row_variable, columns=column_variable, values=color_variable, aggfunc='mean')  # .sort_index().sort_index(axis=1)
            show_colorbar = True
        heatmap_data = heatmap_data.reindex(index=all_rows, columns=all_cols)

        # Build dictionaries for additional variables (used in additional axis labels and hover labels);
        if add_new_figure:
            additional_column_variable_dict = {}
            for col in heatmap_data.columns:
                additional_column_variable_dict[col] = {}
                for var in additional_column_variable:
                    item = df_i.groupby(column_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(col, 'Unknown')
                    additional_column_variable_dict[col][var] = str(item)

            additional_row_variable_dict = {}
            for row in heatmap_data.index:
                additional_row_variable_dict[row] = {}
                for var in additional_row_variable:
                    item = df_i.groupby(row_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(row, 'Unknown')
                    additional_row_variable_dict[row][var] = str(item)
            colorbar_dict = initial_colorbar_dict.copy()
            if global_zaxis_tickvals and global_zaxis_ticktext:
                colorbar_dict.update(tickvals=global_zaxis_tickvals, ticktext=global_zaxis_ticktext)
            z_smooth_option = False if z_smooth_option == 'False' else z_smooth_option

        # if graphs end up being superimposed here (more than one df detected with Row/Column, but no multi_dataframes_id_col specified), the data from last set is on top
        custom_hover_data = np.array([
            [[(f"{chr(64 + int(y))}{f'{x:.0f}' if isinstance(x, (int, float, numbers.Integral)) and not pd.isna(x) else x}" if plate_rows_as_alpha else f"{row_variable}: {int(y)}<br>{column_variable}: {int(x)}"),
                 ('N/A' if pd.isna(z_hover_labels.at[y, x] if isinstance(df_i[color_variable].dtype, CategoricalDtype) else heatmap_data.at[y, x]) else f'{z_hover_labels.at[y, x]:.1f}' \
                     if isinstance(z_hover_labels.at[y, x] if isinstance(df_i[color_variable].dtype, CategoricalDtype) else heatmap_data.at[y, x], (int, float, numbers.Integral)) else \
                     str(z_hover_labels.at[y, x] if isinstance(df_i[color_variable].dtype, CategoricalDtype) else heatmap_data.at[y, x])),]
                + [additional_row_variable_dict.get(y, {}).get(var, '') for var in additional_row_variable if var != color_variable and var not in additional_column_variable]  # we don't want repeat data in the hover labels
                + [additional_column_variable_dict.get(x, {}).get(var, '') for var in additional_column_variable if var != color_variable and var not in additional_row_variable]
                for x in heatmap_data.columns
                ]
            for y in heatmap_data.index
        ])

        # Build hovertemplate string dynamically;
        hovertemplate = '%{customdata[0]}<br>' + f'{color_variable}: %{{customdata[1]}}<br>'
        if additional_row_variable:
            for i, var in enumerate([val for val in additional_row_variable if val != color_variable and val not in additional_column_variable]):
                hovertemplate += f'{var}: %{{customdata[{2 + i}]}}<br>'
        if additional_column_variable:
            start_idx = 2 + len([val for val in additional_row_variable if val != color_variable and val not in additional_column_variable])
            for j, var in enumerate([val for val in additional_column_variable if val != color_variable and val not in additional_row_variable]):
                hovertemplate += f'{var}: %{{customdata[{start_idx + j}]}}<br>'
        hovertemplate += '<extra></extra>'

        # Create heatmap
        if add_new_figure: # new figure
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                customdata=custom_hover_data,
                colorscale=colorscale,
                showscale=show_colorbar,
                colorbar=colorbar_dict if show_colorbar else None,
                zmin=global_zaxis_min,
                zmax=global_zaxis_max,
                zsmooth=z_smooth_option,
                hovertemplate=hovertemplate,
            ))

            # Add a color legend if color_variable is categorical
            if show_colorbar:
                fig.add_annotation(text=color_variable, font=colorbar_titlefont, textangle=0, showarrow=False, xref='paper', yref='paper', x=fig.data[0].colorbar.x + 0.0075, y=fig.data[0].colorbar.y + fig.data[0].colorbar.len, xanchor='left', yanchor='bottom')
            else:
                for category, color in zip(df_i[color_variable].cat.categories, colorscale):
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), legendgroup='color_categories', showlegend=True, name=str(category)))

            # Add axis labels on x-axis, including for additional_column_variable if applicable (columns)
            for i, x_val in enumerate(heatmap_data.columns):
                if additional_column_variable:
                    label_text = f"{f'{x_val:.0f}' if isinstance(x_val, (int, float, numbers.Integral)) and not pd.isna(x_val) else x_val}<br>" + \
                                 '<br>'.join([additional_column_variable_dict.get(x_val, {}).get(var, 'Unknown') for var in additional_column_variable])
                else:
                    label_text = f"{f'{x_val:.0f}' if isinstance(x_val, (int, float, numbers.Integral)) and not pd.isna(x_val) else x_val}"
                fig.add_annotation(x=i, y=-0.01, xref='x', yref='paper', text=label_text, showarrow=False, xanchor='center', yanchor='top', align='center')

            # Add axis labels on y-axis, inluding for additional_row_variable if applicable (rows)
            for i, y_val in enumerate(heatmap_data.index):
                alpha_label = chr(64 + int(y_val)) if plate_rows_as_alpha else int(y_val)
                if additional_row_variable:
                    row_meta = '<br>'.join([additional_row_variable_dict.get(y_val, {}).get(var, 'Unknown') for var in additional_row_variable])
                    label_text = f'{alpha_label}<br>{row_meta}'
                else:
                    label_text = alpha_label
                fig.add_annotation(x=-0.01, y=i, xref='x domain', yref='y', text=label_text, showarrow=False, xanchor='right', yanchor='middle', align='center')

            n_rows, n_cols = heatmap_data.shape
            if n_rows > max_figure_rows:
                max_figure_rows = n_rows
            if n_cols > max_figure_columns:
                max_figure_columns = n_cols

            graph_title_updated = set_graph_title(key, graph_title, split_by_variable, multi_dataframes_id_col, number_of_dfs, graph_index, dfs_delimiter)
            if graph_title_updated:
                fig.add_annotation(text=graph_title_updated, font=graph_titlefont, xref='paper', yref='paper', x=-0.075, y=1.1, xanchor='left', yanchor='top', showarrow=False)
            top_margin = 80 if graph_title_updated not in [None, ''] else 60
            bottom_margin = 90 if not additional_column_variable else 90 + 20 * (len(additional_column_variable))
            xaxis_title_standoff = 40 if not additional_column_variable else 40 + 15 * (len(additional_column_variable))
            yaxis_title_standoff = 40 if not additional_row_variable else 40 + 15 * (len(additional_row_variable))
            fig.update_layout(
                margin=dict(t=top_margin, b=bottom_margin),
                xaxis=dict(title=dict(text=column_variable, font=axis_titlefont), title_standoff=xaxis_title_standoff, type='category', tickmode='array', tickvals=list(heatmap_data.columns), **axis_tickstyle, showticklabels=False, showgrid=False, automargin=True),
                yaxis=dict(title=dict(text=row_variable, font=axis_titlefont), title_standoff=yaxis_title_standoff, type='category', tickmode='array', tickvals=list(heatmap_data.index), autorange='reversed', **axis_tickstyle, showticklabels=False, showgrid=False, automargin=True),
                legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02, font=legend_font, itemsizing='trace'),
                hoverlabel=hoverlabel_font,
                shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1), layer='above')],
                template='plotly_white',
            )
            plots.append(fig)
        else:
            add_trace_to = fig if not split_by_variable else plots[int(i - (len(dfs)//number_of_dfs))]
            add_trace_to.add_trace(go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                customdata=custom_hover_data,
                colorscale=colorscale,
                showscale=show_colorbar,
                colorbar=colorbar_dict if show_colorbar else None,
                zmin=global_zaxis_min,
                zmax=global_zaxis_max,
                zsmooth=z_smooth_option,
                hovertemplate=hovertemplate,
            ))

    return plots, max_figure_rows, max_figure_columns

def generate_piecharts_graph(dfs, row_variable, column_variable,  piecharts_variables, additional_row_variable, additional_column_variable, normalization_method, normalization_value, piechart_donut, \
                             piechart_cakeplots, colorscale, split_by_variable, plate_rows_as_alpha, multi_dataframes_id_col, number_of_dfs, multi_dataframes_reverse, graph_title, plate_variables_columns):

    def compute_pie_values_and_labels(values, labels, normalization_method, normalization_value, well_id, pie_index, number_of_pies, key):
        pie_border = dict(color='black', width=1)
        values = [max(float(val), 0) if not pd.isna(val) else 0 for val in values] # we convert all negative values to 0 here for now
        all_zero = 0

        if number_of_pies > 1: # then we want to include the series name
            well_id = well_id + ' - ' + key
        if int(all(v == 0 for v in values)):
            pie_vals = [1]
            labels = ['Empty']
            hover_info = [f'{well_id}<br>Empty: 0%']
            all_zero = 1
        elif normalization_method == 0:
            total = sum(values) or 1
            pie_vals = [(v / total) * 100 for v in values]
            # changed to values from original_values and updated_labels to labels
            hover_info = [f'{well_id}<br>{label}: {val:.1f}% [{orig:.1f}]' for label, val, orig in
                          zip(labels, pie_vals, values)]
        elif normalization_method == 1:
            pie_vals = values.copy()
            total = sum(pie_vals)
            hover_info = [f'{well_id}<br>{label}: {val:.1f}%' for label, val in zip(labels, pie_vals)]
            if total < 100:
                unknown_percent = 100 - total
                labels.append('Unknown')
                pie_vals.append(unknown_percent)
                hover_info.append(f'{well_id}<br>Unknown: {unknown_percent:.1f}%')
            elif total > 100:
                pie_border = dict(color='red', width=2)
        elif normalization_method == 2:
            max_val = normalization_value or 1
            pie_vals = [(v / max_val) * 100 for v in values]
            hover_info = [f'{well_id}<br>{label}: {val:.1f}% [{(val * max_val / 100):.3g}]' for label, val in
                          zip(labels, pie_vals)]
            total = sum(pie_vals)
            if total < 100:
                unknown_percent = 100 - total
                unknown_value = max_val * (unknown_percent / 100)
                labels.append('Unknown')
                pie_vals.append(unknown_percent)
                hover_info.append(f'{well_id}<br>Unknown: {unknown_percent:.1f}% [{unknown_value:.1f}]')
            elif total > 100:
                pie_border = dict(color='red', width=2)
        else:
            pie_vals = [1]
            labels = ['Invalid normalization method']
            hover_info = [f'{well_id}<br>Error: invalid normalization method']

        show_legend = True
        if pie_border['color'] != 'red' and pie_index > 0:
            pie_border = dict(color='lightslategrey', width=1)

        filtered_vals = [(val, lab, hov) for val, lab, hov in zip(pie_vals, labels, hover_info) if val > 0] # here we remove 0 values so the info doesn't show in the pie

        if filtered_vals:
            pie_vals, labels, hover_info = zip(*filtered_vals)
            pie_vals, labels, hover_info = list(pie_vals), list(labels), list(hover_info)


        return pie_vals, labels, hover_info, pie_border, all_zero, show_legend

    for key, df_i in dfs.items():
        dfs[key] = df_i[df_i[row_variable].notna() & df_i[column_variable].notna() & (df_i[row_variable].astype(str).str.strip() != '') & (df_i[column_variable].astype(str).str.strip() != '')].copy()
        dfs[key][row_variable] = pd.to_numeric(dfs[key][row_variable], errors='coerce')
        dfs[key][column_variable] = pd.to_numeric(dfs[key][column_variable], errors='coerce')

    dfs_combined = pd.concat(dfs.values(), ignore_index=True)
    dfs_combined = dfs_combined[dfs_combined[row_variable].notna() & dfs_combined[column_variable].notna() & (dfs_combined[row_variable].astype(str).str.strip() != '') & (dfs_combined[column_variable].astype(str).str.strip() != '')].copy()
    selected_slice_colors_from_colorscale = get_discrete_colorscale(colorscale, len(piecharts_variables))
    piecharts_variables_colors = dict(zip(piecharts_variables, get_discrete_colorscale(colorscale, len(piecharts_variables))))
    piecharts_variables_colors['Unknown'] = 'white'
    base_unit, spacing_px, piechart_hole_size, piechart_shadow, piechart_shadow_x_offset, piechart_shadow_y_offset, piechart_shadow_add_size = 100, 10, 0.3, True, 0.0025, 0.006, 0.005,

    if multi_dataframes_id_col and piechart_cakeplots and number_of_dfs > 1:
        number_of_pies = number_of_dfs
    else:
        number_of_pies = 1

    # <editor-fold desc="*Placeholder in case*">
    # ****leave for now, but disabled currently - will plot multiple dataframes on separate arrays, unless cake_plots
    #dfs_grouped = {}
    #if split_by_variable and multi_dataframes_id_col:
    #    #  Case 4: split by both
    #    if piechart_cakeplots:
    #        for key, df_i in dfs.items():
    #            series_name, split_var = key.split(' - ', 1)
    #            dfs_grouped.setdefault(split_var, {})
    #            dfs_grouped[split_var][series_name] = df_i
    #    else:
    #        for key, df_i in dfs.items():
    #            dfs_grouped.setdefault(key, {})
    #            dfs_grouped[key]['All'] = df_i
    #elif split_by_variable: #**** issue here if more than one df and if cake_plots
    #    # Case 2: split only by variable
    #    if number_of_dfs == 1:
    #        for key, df_i in dfs.items():
    #            split_var = key.replace('All - ', '') if key.startswith('All - ') else key
    #            dfs_grouped.setdefault(split_var, {})
    #            dfs_grouped[split_var]['All'] = df_i
    #    else:
    #        for key, df_i in dfs.items():
    #            series_name, split_var = key.split(' - ', 1)
    #            dfs_grouped.setdefault(split_var, {})
    #            dfs_grouped[split_var][series_name] = df_i
    #elif multi_dataframes_id_col:
    #    # Case 3: split only by series
    #    if piechart_cakeplots:
    #        dfs_grouped['All'] = dfs
    #    else:
    #        for key, df_i in dfs.items():
    #            dfs_grouped.setdefault(key, {})
    #            dfs_grouped[key]['All'] = df_i
    #elif number_of_dfs > 1:
    #    dfs_grouped['All'] = dfs
    #else:
    #    # Case 1: no split at all
    #    dfs_grouped['All'] = {'All': list(dfs.values())[0]}
    #< / editor - fold >

    dfs_grouped = {} # necessary with the current code but may be simplified in the future
    if split_by_variable:
        if piechart_cakeplots:
            for key, df_i in dfs.items():
                series_name, split_var = key.split(dfs_delimiter, 1)
                dfs_grouped.setdefault(split_var, {})[series_name] = df_i
        else:
            for key, df_i in dfs.items():
                dfs_grouped.setdefault(key, {})['All'] = df_i
    else:
        if piechart_cakeplots:
            dfs_grouped['All'] = dfs
        else:
            for key, df_i in dfs.items():
                dfs_grouped.setdefault(key, {})['All'] = df_i

    # Build maps for the additional row/column variables
    global_plate_rows, global_plate_columns = list(range(1, int(dfs_combined[row_variable].max()) + 1)), list(range(1, int(dfs_combined[column_variable].max()) + 1))
    additional_column_variable_hover_map, additional_column_variable_axis_map = {}, {}
    if additional_column_variable:
        if not isinstance(additional_column_variable, list):
            additional_column_variable = [additional_column_variable]
        for col in global_plate_columns:
            hover_values = []
            axis_values = []
            for var in additional_column_variable:
                item = dfs_combined.groupby(column_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(col, 'Unknown')
                axis_values.append(str(item))
                if not any(var in hv for hv in hover_values):
                    hover_values.append(f'{var}: {item}')
            additional_column_variable_hover_map[col] = '<br>'.join(hover_values)
            additional_column_variable_axis_map[col] = '<br>'.join(axis_values)

    additional_row_variable_hover_map, additional_row_variable_axis_map = {}, {}
    if additional_row_variable:
        if not isinstance(additional_row_variable, list):
            additional_row_variable = [additional_row_variable]
        for row in global_plate_rows:
            hover_values = []
            axis_values = []
            for var in additional_row_variable:
                item = dfs_combined.groupby(row_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(row, 'Unknown')
                axis_values.append(str(item))
                if not any(var in hv for hv in hover_values):
                    hover_values.append(f'{var}: {item}')
            additional_row_variable_hover_map[row] = '<br>'.join(hover_values)
            additional_row_variable_axis_map[row] = '<br>'.join(axis_values)

    # Generate pie traces manually
    plots = []
    max_figure_columns, max_figure_rows = 0, 0
    for graph_index, (group_name, series_dict) in enumerate(dfs_grouped.items()):
        plate_rows = sorted({value for df_i in series_dict.values() for value in df_i[row_variable].dropna().unique()})
        plate_columns = sorted({value for df_i in series_dict.values() for value in df_i[column_variable].dropna().unique()})
        max_figure_rows = len(plate_rows) if len(plate_rows) > max_figure_rows else max_figure_rows
        max_figure_columns = len(plate_columns) if len(plate_columns) > max_figure_columns else max_figure_columns
        plot_width, plot_height = base_unit * (len(plate_columns)) + ((len(plate_columns)) - 1) * spacing_px, base_unit * (len(plate_rows)) + ((len(plate_rows)) - 1) * spacing_px
        spacing_x, spacing_y, cell_width, cell_height = spacing_px / plot_width, spacing_px / plot_height, 1 / (len(plate_columns)), 1 / (len(plate_rows))
        pie_width, pie_height = cell_width - spacing_x, cell_height - spacing_y

        fig = go.Figure()
        for row_index, row in enumerate(plate_rows):  # This iterates through each element of the array, providing the 0-based index and the value
            for col_index, col in enumerate(plate_columns):
                x_center = (col_index + 0.5) * cell_width
                y_center = 1 - (row_index + 0.5) * cell_height

                # Dealing with additional variables for hover_labels
                row_meta = additional_row_variable_hover_map.get(row, '') if additional_row_variable else ''
                col_meta = additional_column_variable_hover_map.get(col, '') if additional_column_variable else ''
                well_label = f'{chr(64 + int(row))}{int(col)}' if plate_rows_as_alpha else f'{row_variable}: {int(row)}<br>{column_variable}: {int(col)}'

                for series_index, (key, df_i) in enumerate(series_dict.items()):
                    scale_factor = 1/((series_index/2)+1) if number_of_pies > 1 else 1
                    pie_domain = dict(x=[x_center - (pie_width*scale_factor) / 2, x_center + (pie_width*scale_factor) / 2], y=[y_center - (pie_height*scale_factor) / 2, y_center + (pie_height*scale_factor) / 2])
                    if piechart_shadow:
                        pie_domain_shadow = dict(
                            x=[max(0, pie_domain['x'][0] - (pie_domain['x'][1] - pie_domain['x'][0]) * piechart_shadow_add_size + piechart_shadow_x_offset),
                            min(1, pie_domain['x'][1] + (pie_domain['x'][1] - pie_domain['x'][0]) * piechart_shadow_add_size + piechart_shadow_x_offset)],
                            y=[max(0, pie_domain['y'][0] - (pie_domain['y'][1] - pie_domain['y'][0]) * piechart_shadow_add_size - piechart_shadow_y_offset),
                            min(1, pie_domain['y'][1] + (pie_domain['y'][1] - pie_domain['y'][0]) * piechart_shadow_add_size - piechart_shadow_y_offset)]
                        )
                    row_data = df_i[(df_i[row_variable] == row) & (df_i[column_variable] == col)]
                    if not row_data.empty:
                        row_values = [row_data.iloc[0][var] for var in piecharts_variables]
                        updated_vals, updated_labels, hover_text, updated_pie_border, all_zero, show_legend = compute_pie_values_and_labels(row_values, piecharts_variables.copy(), \
                                                                                                                normalization_method, normalization_value, well_label, series_index, number_of_pies, key)
                        if col_meta or row_meta:
                            meta_text = f"{row_meta}{'<br>' if row_meta and col_meta else ''}{col_meta}"
                            hover_text = [f'{ht}<br>{meta_text}' for ht in hover_text]
                        updated_colors = [piecharts_variables_colors[label] for label in updated_labels] + ['white'] * (len(updated_labels) - len(selected_slice_colors_from_colorscale)) if all_zero == 0 else ['white']

                        if number_of_pies > 1:
                            color_shift_factor = (series_index / number_of_pies) / 2
                        else:
                            color_shift_factor = 0

                        # Add a pattern to the unknown slices and fix the colour;
                        marker_patterns = ['.' if lab == 'Unknown' else '' for lab in updated_labels]
                        updated_colors = ['palegoldenrod' if lab == 'Unknown' else col for lab, col in zip(updated_labels, updated_colors)]

                        # Lighten the inner pies some amount
                        updated_colors = [lighten_rgb(c, color_shift_factor) for c in updated_colors]
                        pie_args = {'values': updated_vals, 'labels': updated_labels, 'marker': dict(colors=updated_colors, line=updated_pie_border, pattern=dict(shape=marker_patterns)), \
                                    'domain': pie_domain, 'text': hover_text, 'hoverinfo': 'text', 'textinfo': 'none', 'sort': False, 'name':key, 'legendgroup':'S' + str(series_index + 1), \
                                    'showlegend': show_legend, 'legendrank': 2*(series_index+1), **({'hole':piechart_hole_size} if piechart_donut else {})}
                    #else: # taken off for now since was an issue when using split_by_variable
                    #    updated_labels, show_legend = (['Missing'], True) if series_index == 0 else (None, False)
                    #    pie_args = {'values': [1], 'labels': updated_labels,'marker': dict(colors=['lightgrey'], pattern=dict(shape=['x']), line=dict(color='black', width=1)), 'domain': pie_domain, 'hoverinfo': 'skip', \
                    #        'textinfo': 'none', 'name':key, 'legendgroup':'S' + str(series_index + 1), 'showlegend':show_legend, 'legendrank': 2*(series_index+1), **({'hole': piechart_hole_size} if piechart_donut else {})}

                        if piechart_shadow:
                            pie_args_shadow = {'values': [1], 'labels': None, 'marker': dict(colors=['lightgrey'], line=None), 'domain': pie_domain_shadow, 'showlegend': False, \
                                            'hoverinfo': 'skip', 'textinfo': 'none',  **({'hole': piechart_hole_size} if piechart_donut else {})}
                        if series_index == 0:
                            fig.add_trace(go.Pie(**pie_args_shadow))
                        fig.add_trace(go.Pie(**pie_args))

        # Add dummy pies to add titles to some of the legend; unfortunately can't align them differently
        if number_of_pies > 1:
            for series_index, key in enumerate(series_dict.keys()):
                dummy_trace_domain = dict(
                                x = [(pie_domain['x'][0] + ((pie_domain['x'][1] - pie_domain['x'][0])/2)) - 0.001, (pie_domain['x'][0] + ((pie_domain['x'][1] - pie_domain['x'][0])/2)) + 0.001],
                                y = [(pie_domain['y'][0] + ((pie_domain['y'][1] - pie_domain['y'][0])/2)) - 0.001, (pie_domain['y'][0] + ((pie_domain['y'][1] - pie_domain['y'][0])/2)) + 0.001]
                                )
                fig.add_trace(go.Pie(values=[1], labels=[key], marker=dict(colors=['rgba(255,255,255,0)'], line=None), domain=dummy_trace_domain, showlegend=True, \
                            legendgroup=f'S{series_index + 1}', legendrank=1 + (series_index * 2), hoverinfo='skip', textinfo='none'))

        # Add axis labels on x-axis (columns);
        column_variable_label_y = 0 - 0.05 - (len(additional_column_variable) if additional_column_variable else 0)*0.05 # was previously -0.08

        fig.add_annotation(x=0.5, y=column_variable_label_y, text=column_variable, showarrow=False, xref='paper', yref='paper', font=axis_titlefont, xanchor='center', yanchor='top')
        for col_index, col in enumerate(plate_columns):
            if any(col in df_i[column_variable].values for df_i in series_dict.values()):
                x = (col_index + 0.5) * cell_width
                label_text = f"{int(col)}<br>{additional_column_variable_axis_map.get(col, 'Unknown')}" if additional_column_variable else str(int(col))
                fig.add_shape(type='line', xref='paper', yref='paper', x0=x, x1=x, y0=0, y1=-0.01, line=dict(color='black', width=1), fillcolor='rgba(0,0,0,0)')
                fig.add_annotation(x=x, y=-0.01, text=label_text, showarrow=False, xref='paper', yref='paper', font=axis_tickfont, xanchor='center', yanchor='top')

        # Add axis labels on y-axis (rows);
        fig.add_annotation(x=-0.06, y=0.5, text=row_variable, showarrow=False, xref='paper', yref='paper', font=axis_titlefont, xanchor='right', yanchor='middle', textangle=270)
        for row_index, row in enumerate(plate_rows):
            if any(row in df_i[row_variable].values for df_i in series_dict.values()):
                y = 1 - (row_index + 0.5) * cell_height
                label_text = chr(64 + int(row)) if plate_rows_as_alpha else int(row)
                if additional_row_variable:
                    label_text += f"<br>{additional_row_variable_axis_map.get(row, 'Unknown')}"
                fig.add_shape(type='line', xref='paper', yref='paper', x0=0, x1=-0.005, y0=y, y1=y, line=dict(color='black', width=1), fillcolor='rgba(0,0,0,0)')
                fig.add_annotation(x=-0.01, y=y, text=label_text, showarrow=False, xref='paper', yref='paper', font=axis_tickfont, xanchor='right', yanchor='middle')

        # Update layout
        if not split_by_variable:
            group_name = ', '.join(series_dict.keys()) if piechart_cakeplots else group_name
        else:
            group_name = ', '.join(series_dict.keys()) + dfs_delimiter + group_name if piechart_cakeplots else group_name
        graph_title_updated = set_graph_title(group_name, graph_title, split_by_variable, multi_dataframes_id_col, number_of_dfs, graph_index, dfs_delimiter)
        if graph_title_updated:
            fig.add_annotation(text=graph_title_updated, font=graph_titlefont, xref='paper', yref='paper', x=-0.075, y=1.075, xanchor='left', yanchor='top', showarrow=False)
        top_margin = 80 if graph_title_updated not in [None, ''] else 60
        bottom_margin = 90 if not additional_column_variable else 90 + 20*(len(additional_column_variable))
        fig.update_layout(
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor='left', font=legend_font, itemsizing='constant', tracegroupgap=15, yanchor='top', bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0)'),
            margin=dict(t=top_margin, b=bottom_margin, l=130),
            hoverlabel=hoverlabel_font,
            shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=2), layer='above')], # add rectangle around pie charts to create plate layout
        )
        plots.append(fig)

    return plots, max_figure_rows, max_figure_columns


def generate_barchart_graph(dfs, barchart_x, barchart_vars, barchart_pattern, barchart_group_vars_by, barchart_barmode_option, colorscale, split_by_variable, plate_rows_as_alpha, \
                            multi_dataframes_id_col, multi_dataframes_reverse, number_of_dfs, graph_title, category_suffix, plate_variables_columns):

    multi_group_by, barmode, yaxis_tickstep, barchart_shadow = barchart_x, barchart_barmode_option.lower(), 10, True
    dfs_combined = pd.concat(dfs.values(), ignore_index=True)
    selected_colors_from_colorscale = get_discrete_colorscale(colorscale, len(barchart_vars))
    #dup_max_x = dfs_combined[barchart_x].value_counts().max() # hold for now

    if barchart_pattern:
        global_marker_pattern_map = {item: plotly_all_marker_patterns[i % len(plotly_all_marker_patterns)] for i, item in enumerate(dfs_combined[barchart_pattern].unique())}
        global_marker_pattern_assigned = [global_marker_pattern_map[val] for val in dfs_combined[barchart_pattern]]

    for key, subdf in dfs.items():
        if is_numeric_dtype(dfs_combined[barchart_x]):
            dfs[key][barchart_x] = subdf[barchart_x].fillna(0).copy()  # currently we convert those rows to 0 from NaN, but do we want to drop them instead?
        for barchart_var in barchart_vars:
            if is_numeric_dtype(dfs_combined[barchart_var]):
                dfs[key][barchart_var] = subdf[barchart_var].fillna(0).copy()
        if barchart_pattern:
            subdf['bar_pattern'] = subdf[barchart_pattern].map(global_marker_pattern_map)

    # We create groups here and treat multi_dataframes a bit differently due to the way Plotly barchart graphs handle multiple values for a data point; if multi_dataframes_id_col is specified then
    # it will plot all series on the same graph(s); otherwise duplicates will be stacked or overlayed existing ones.  For separate graphs by multi_dataframes_id_col, one should add a Series column
    # to the dataframe and use the split_by_variable in order to create them - may address this differently in the future
    dfs_grouped = {}
    if split_by_variable and multi_dataframes_id_col:  #{'F1':{'Series 1': [0-22], 'Series 2': [24-46]}, 'F2': {'Series 1': [1-23], 'Series 2': [25-47]}} # group/split by both
        for key, df_i in dfs.items():
            series_name, split_var = key.split(dfs_delimiter, 1)
            dfs_grouped.setdefault(split_var, {})
            dfs_grouped[split_var][series_name] = df_i
    elif split_by_variable: # {'F1': {'All': [0-22]}, 'F2': {'All': [1-23]}} # group only with split_by_variable
        for key, df_i in dfs.items():
            split_var = key.replace('All' + dfs_delimiter, '') if key.startswith('All' + dfs_delimiter) else key
            dfs_grouped.setdefault(split_var, {})
            dfs_grouped[split_var]['All'] = df_i
    elif multi_dataframes_id_col: # {'All': {'Series1': [0-23], 'Series2': [24-47]}} # group only by dfs as specified by multi_dataframes_id_col
        dfs_grouped['All'] = dfs
    else: # {'All':{'All': [0-23]}} # no grouping
        dfs_grouped['All'] = {'All': list(dfs.values())[0]}

    if isinstance(dfs_combined[barchart_x].dtype, CategoricalDtype):
        barchart_x = barchart_x + '_encoded'
    if isinstance(dfs_combined[barchart_vars[0]].dtype, CategoricalDtype): # we're forcing only numerical here so this is useless for now
        global_yaxis_min = dfs_combined[barchart_vars[0] + '_encoded'].min()
        global_yaxis_max = dfs_combined[barchart_vars[0] + '_encoded'].max()
        for i, var in enumerate(barchart_vars):
            barchart_vars[i] = var + '_encoded'
    else:
        # All graphs should have the same scale and over ALL the barchart_vars
        yaxis_min = min([dfs_combined[var].min() for var in barchart_vars])
        yaxis_max = max([dfs_combined[var].max() for var in barchart_vars])
        if barmode == 'stack':
            max_vals = []
            if number_of_dfs == 1:
                for i, df in dfs.items():
                    max_vals.append(df[barchart_vars].sum(axis=1).max())
            else:
                for var in barchart_vars:
                    max_vals.append(pd.concat([df[[barchart_x, var]] for df in dfs.values()]).groupby(barchart_x)[var].sum().max())
            yaxis_max = max(max_vals)
        global_yaxis_dict, global_yaxis_min, global_yaxis_max, _, _, global_yaxis_dtick = set_axis_arguments(dfs_combined, barchart_vars[0], axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, is_y_variable=True, min_value=yaxis_min, max_value=yaxis_max)

    yaxis_title, colorbar_title = barchart_vars[0] if len(barchart_vars) == 1 else ', '.join(barchart_vars), barchart_vars[0] if len(barchart_vars) == 1 else 'Y-Axis'

    plots = []
    for graph_index, (key, dfs_group) in enumerate(dfs_grouped.items()):
        fig = go.Figure()
        for series_index, (series_key, df_i) in enumerate(dfs_group.items()):
            if barchart_group_vars_by and barchart_group_vars_by != split_by_variable:
                df_i_sorted = df_i.sort_values(by=[barchart_group_vars_by], kind='stable')
                number_per_group = df_i_sorted[barchart_group_vars_by].value_counts(sort=False)
                if number_of_dfs > 1 and series_index != 0:
                    consistent_groups = all(dfi[barchart_group_vars_by].value_counts(sort=False).equals(number_per_group) for dfi in dfs_group.values())
                    if not consistent_groups:
                        barchart_group_vars_by, number_per_group = None, None
            else:
                df_i_sorted = df_i
                number_per_group = None

            color_shift_factor = (series_index / (number_of_dfs if number_of_dfs > 1 else 1)) / 2 if not multi_dataframes_reverse else (series_index / (number_of_dfs if number_of_dfs > 1 else 1))
            updated_colors = [lighten_rgb(c, color_shift_factor) for c in selected_colors_from_colorscale]

            for j, var in enumerate(barchart_vars):
                y_values = df_i_sorted[var]
                if number_of_dfs == 1 and len(barchart_vars) == 1:
                    marker_dict = dict(
                        color=y_values,
                        colorscale=colorscale,
                        colorbar=initial_colorbar_dict.copy(),
                        cmin=global_yaxis_min,
                        cmax=global_yaxis_max
                    )
                else:
                    marker_dict = dict(color=updated_colors[j % len(updated_colors)])
                if barchart_pattern:
                    marker_dict.update(pattern=dict(shape=df_i_sorted['bar_pattern']))

                # Build hovertemplate
                hover_template = ((f'Series {series_index + 1}<br>' if number_of_dfs > 1 else '') + f'{barchart_x}: %{{x}}<br>{var}: %{{y}}')
                if barchart_group_vars_by:
                    hover_template += f'<br>{barchart_group_vars_by}: %{{customdata[0]}}'
                hover_template += '<extra>DF</extra>' if number_of_dfs > 1 else '<extra></extra>'

                if barmode == 'stack':
                    offsetgroup_val = 0 if number_of_dfs == 1 else var
                elif barmode == 'overlay': # and number_of_dfs > 1:
                    offsetgroup_val = j if number_of_dfs > 1 else None
                #elif barmode == 'overlay' and number_of_dfs == 1: #and not any(df[barchart_x].duplicated().any() for df in dfs.values()):
                #    offsetgroup_val = None
                else:
                    offsetgroup_val = f'{series_index}-{j}'

                # Add trace
                fig.add_trace(go.Bar(
                    x=df_i_sorted[barchart_x],
                    y=y_values,
                    marker=marker_dict,
                    name=var,
                    offsetgroup=offsetgroup_val,
                    legendgroup=f'S{series_index + 1}',
                    legendrank=2 * (series_index + 1) if number_of_dfs > 1 else None,
                    customdata=df_i_sorted[[barchart_group_vars_by]].to_numpy() if barchart_group_vars_by else None,
                    hovertemplate=hover_template,
                    showlegend=True,
                    #showlegend=False if number_of_dfs == 1 else True
                ))
            if number_of_dfs > 1:  # adding legends for multiple series (titles)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=0, color='rgba(0,0,0,0)'), name=series_key, legendgroup='S' + str(series_index + 1), \
                                legendrank=1 + (series_index * 2), showlegend=True))

        # Add labels and brackets for the barchart_group_vars_by
        if barchart_group_vars_by and number_per_group is not None:
            total_bars = len(barchart_vars) * len(df_i_sorted[barchart_x]) * number_of_dfs  # if multi_dataframes_id_col else len(barchart_vars)*len(df_i_sorted[barchart_x])*number_of_dfs
            group_ranges = {}
            current_pos = 0
            for group_value, count_per_group in number_per_group.items(): # total bars for this group (consider multiple Y variables)
                bars_in_group = count_per_group * len(barchart_vars) * number_of_dfs
                group_ranges[group_value] = (current_pos, current_pos + bars_in_group - 1)
                current_pos += bars_in_group
            for group_value, (start_idx, end_idx) in group_ranges.items():
                x_center = (start_idx + end_idx + 1) / 2 / total_bars
                x_start = start_idx / total_bars
                x_end = (end_idx + 1) / total_bars
                y_bracket = -0.06
                fig.add_shape(type='line', xref='paper', yref='paper', x0=x_start, y0=y_bracket, x1=x_end, y1=y_bracket, line=dict(color='black', width=1))  # bracket
                tick_height = -0.01
                for x in [x_start, x_end]:
                    fig.add_shape(type='line', xref='paper', yref='paper', x0=x, y0=y_bracket, x1=x, y1=y_bracket - tick_height, line=dict(color='black', width=1))  # tick marks
                fig.add_annotation(x=x_center, y=-0.08, text=str(group_value), showarrow=False, xref='paper', yref='paper', font=axis_tickfont, xanchor='center', yanchor='top')  # group value

        if barchart_pattern: # and series_index + 1 == number_of_dfs: # adding a legend for the patterns
            for item, pattern in global_marker_pattern_map.items():
                fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color='lightgray', pattern=dict(shape=pattern, fillmode='overlay')), name=item, legendgroup='pattern', showlegend=True))

        if hasattr(fig.data[0].marker, 'colorbar') and fig.data[0].marker.colorbar is not None:
            if len(fig.data[0].marker.colorbar.to_plotly_json()) > 0:
                fig.add_annotation(text=colorbar_title, font=colorbar_titlefont, textangle=0, showarrow=False, xref='paper', yref='paper', x=fig.data[0].marker.colorbar.x + 0.0075, y=fig.data[0].marker.colorbar.y + fig.data[0].marker.colorbar.len, xanchor='left', yanchor='bottom')

        graph_title_updated = set_graph_title((series_key + dfs_delimiter + key), graph_title, split_by_variable, multi_dataframes_id_col, 1, graph_index, dfs_delimiter) # forcing number_of_dfs to 1 for graph title due to the way Barchart is set up
        if graph_title_updated:
            fig.add_annotation(text=graph_title_updated, font=graph_titlefont, xref='paper', yref='paper', x=-0.065, y=1.1, xanchor='left', yanchor='top', showarrow=False)
        top_margin = 80 if graph_title_updated not in [None, ''] else 60
        fig.update_layout(
            barmode=barmode,
            legend=dict(orientation='h', yanchor='top', y=-0.15 if barchart_group_vars_by else -0.1, xanchor='right', x=1, font=legend_font),
            margin=dict(t=top_margin, b=10),
            xaxis=dict(showline=True, linecolor='black', mirror=True, title=dict(text=barchart_x, font=axis_titlefont, standoff=55 if barchart_group_vars_by else 15), type='category', categoryorder='array', \
                    categoryarray=df_i_sorted[barchart_x], **axis_tickstyle, showgrid=False, gridcolor='white', dtick=1),
            yaxis=dict(showline=True, linecolor='black', mirror=True, title=dict(text=yaxis_title, font=axis_titlefont), **global_yaxis_dict, showgrid=False, gridcolor='white'),
            plot_bgcolor='white',
            hoverlabel=hoverlabel_font,
        )

        plots.append(fig)

    return plots

def generate_dumbbell_graph(dfs, x_variable, y_variable, color_variable, symbol_variable, x_grouped_over_vars, colorscale, split_by_variable, plate_rows_as_alpha, multi_dataframes_id_col, \
                            number_of_dfs, graph_title, category_suffix, plate_variables_columns):
    yaxis_tickstep = 10
    dfs_combined = pd.concat(dfs.values(), ignore_index=True)

    # Fill the numeric columns with 0 for na; currently only y_variable can be numeric;
    for key, subdf in dfs.items():
        if is_numeric_dtype(dfs_combined[y_variable]):
            dfs[key][y_variable] = subdf[y_variable].fillna(0).copy()

    global_marker_color_map, global_marker_symbol_map = None, None
    if color_variable:
        global_marker_color_map = {
            color: get_discrete_colorscale(colorscale, len(dfs_combined[color_variable].dropna().unique()))[i % len(get_discrete_colorscale(colorscale, len(dfs_combined[color_variable].dropna().unique())))]
            for i, color in enumerate(dfs_combined[color_variable].dropna().unique())
            }
    if symbol_variable:
        global_marker_symbol_map = {item: marker_symbols[i % len(marker_symbols)] for i, item in enumerate(dfs_combined[symbol_variable].unique())}

    global_yaxis_dict = set_axis_arguments(dfs_combined, y_variable, axis_tickstep, plate_rows_as_alpha, plate_variables_columns, axis_tickstyle, axis_padding=0.5)[0]
    vars_to_group_over = list(dict.fromkeys([v for v in [color_variable, symbol_variable] + (x_grouped_over_vars or []) if v is not None]))

    plots = []
    for graph_index, (key, df_i) in enumerate(dfs.items()):
        fig = go.Figure()
        # Group by specified variables first - each one gets a dumbbell
        for group_keys, subdf in df_i.groupby(vars_to_group_over, observed=False):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            group_labels = '<br>'.join([f"{var}: {val}" for var, val in zip(vars_to_group_over, group_keys)])

            line_color = line_color = global_marker_color_map[subdf[color_variable].iloc[0]] if (color_variable and global_marker_color_map) else 'black'
            fig.add_trace(go.Scatter( # this adds the lines
                x=subdf[x_variable],
                y=subdf[y_variable],
                mode='lines',
                line = dict(color=line_color, width=1),
                showlegend=False,
            ))

            for _, row in subdf.iterrows():
                marker_color = global_marker_color_map[row[color_variable]] if (color_variable and global_marker_color_map) else default_single_marker['color']
                marker_symbol = global_marker_symbol_map[row[symbol_variable]] if (symbol_variable and global_marker_symbol_map) else default_single_marker['symbol']
                hover_template = f'{y_variable}: {row[y_variable]:.1f}<br>{x_variable}: {row[x_variable]}<br>{group_labels}<extra></extra>'
                fig.add_trace(go.Scatter(
                    x = [row[x_variable]],
                    y = [row[y_variable]],
                    mode = 'markers',
                    marker=dict(size=12, color=marker_color, symbol=marker_symbol, line=dict(color='black', width=1)),
                    hovertemplate=hover_template,
                    showlegend = False  # we'll add legend manually
             ))

        # Add legend entries (color/symbol variables)
        if color_variable and global_marker_color_map:
            for color_var, color in global_marker_color_map.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=12, color=color, symbol='circle'),
                    name=f'{color_variable}: {color_var}'
                ))
        if symbol_variable and global_marker_symbol_map:
            for symbol_var, symbol in global_marker_symbol_map.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=12, color='lightgray', symbol=symbol, line=dict(color='black', width=1)),
                    name=f'{symbol_variable}: {symbol_var}'
                 ))

        graph_title_updated = set_graph_title(key, graph_title, split_by_variable, multi_dataframes_id_col, number_of_dfs, graph_index, dfs_delimiter)
        if graph_title_updated:
            fig.add_annotation(text=graph_title_updated, font=graph_titlefont, xref='paper', yref='paper', x=-0.075, y=1.1, xanchor='left', yanchor='top', showarrow=False)
        top_margin = 110 if graph_title_updated not in [None, ''] else 60
        xaxis_title = x_variable + '<br>' + '[' + ', '.join(vars_to_group_over) + ']'
        fig.update_layout(
            margin=dict(t=top_margin),
            xaxis=dict(title=dict(text=xaxis_title, font=axis_titlefont, standoff=30), showline=True, linecolor='black', mirror=True, type='category', **axis_tickstyle, showgrid=False, gridcolor='white'),
            yaxis=dict(title=dict(text=y_variable, font=axis_titlefont, standoff=30), showline=True, linecolor='black', mirror=True, **global_yaxis_dict, showgrid=False, gridcolor='white'),
            template='plotly_white',
            hoverlabel=hoverlabel_font,
            legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02, font=legend_font, itemsizing='trace'),
        )
        plots.append(fig)

    return plots






