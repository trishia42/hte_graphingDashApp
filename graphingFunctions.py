import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype, is_object_dtype
#import plotly.express as px
import plotly.graph_objects as go
#import plotly.colors as pc
import math
import numpy as np
import numbers
from miscellaneousFunctions import *

def generate_parallel_coordinates_graph(df, parallel_variables, color_variable, color_scale, graph_title, category_suffix):

    color_tick_step, variable_tick_step = 10, 5 # arbitrary here for now
    if color_variable not in [None, 'None']:
        df[color_variable] = df[color_variable].fillna(0).copy() # currently we convert those rows to 0 from NaN, but do we want to drop them instead?
    plot_dimensions = []

    for col in parallel_variables:
        print(col)
        if isinstance(df[col].dtype, CategoricalDtype):
            categories = df[col].cat.categories
            plot_dimensions.append(dict(range=[1, len(categories)], tickvals=list(range(1, len(categories) + 1)), ticktext=list(categories), label=col, values=df[col + category_suffix]))
        elif is_numeric_dtype(df[col]) and not col.endswith(category_suffix):
            tickstep = color_tick_step if col == color_variable else variable_tick_step
            cmin, cmax = 0, math.ceil(df[col].max() / tickstep) * tickstep
            tickvals, val = [], cmin
            while val < cmax:
                tickvals.append(val)
                val += tickstep
            tickvals.append(cmax)
            #f col == color_variable:
            #    tickvals_colorbar = tickvals
            #    print(tickvals_colorbar)
            plot_dimensions.append(dict(range=[cmin, cmax], tickvals=tickvals, label=col.title(), values=df[col]))

        line_dict = {}
        if color_variable not in ['None', None]:
            #line_dict = dict(color=df[color_variable], colorscale=color_scale, showscale=True, colorbar=dict(title=color_variable, dtick=color_tick_step, len=1.08, y=0.5, yanchor='middle', x=1.02, xanchor='left'), \
            #                cmin=0, cmax=math.ceil(df[color_variable].max() / color_tick_step) * color_tick_step)
            line_dict = dict(color=df[color_variable], colorscale=color_scale, showscale=True, colorbar=dict(title=color_variable, dtick=color_tick_step, y=-0.015, yanchor='bottom', x=1.05, xanchor='left', \
                            len=1.06), cmin=0, cmax=math.ceil(df[color_variable].max() / color_tick_step) * color_tick_step)

        fig = go.Figure(data=go.Parcoords(
            line=line_dict,
            dimensions=plot_dimensions,
            unselected=dict(line=dict(color='lightgray', opacity=0.1)),
            labelfont=dict(color='darkblue', family='Times New Roman', size=20, weight='bold'),
            tickfont=dict(color='black', family='Times New Roman', size=16, weight='bold'),
        ))
        top_margin = 140 if graph_title not in [None, ""] else 60
        fig.update_layout(
            title=dict(text=graph_title, font=dict(color='black', family='Times New Roman', size=26, weight='bold'), x=0, xanchor='left', y=0.97, yanchor='top'),
            margin=(dict(t=top_margin)) #this controls tha padding around the entire graph area, including the title and axes
        )

    return fig

def generate_scatter_bubble_graph(df, x_variable, y_variable, size_variable, symbol_variable, color_variable, color_scale, graph_title, category_suffix):

    def add_hover_and_custom_data(df_or_series, custom_data, hover_lines, insert_or_append, col=None, is_numeric=False, insert_index=0, fmt=":.2f", hover_label=None, added_labels=None):
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
                hover_lines.append(f"{label}: %{{customdata[{len(custom_data) - 1}]{fmt if is_numeric else ''}}}")
            else:
                hover_lines.append(f"%{{customdata[{len(custom_data) - 1}]{fmt if is_numeric else ''}}}")
        else:  # insert
            custom_data.insert(insert_index, series)
            if label.lower() != 'well_label':
                hover_lines.insert(insert_index, f"{label}: %{{customdata[{insert_index}]{fmt if is_numeric else ''}}}")
            else:
                hover_lines.insert(insert_index,f"%{{customdata[{insert_index}]{fmt if is_numeric else ''}}}")
        added_labels.add(label)

    # Set some defaults
    default_single_marker, zero_value_marker = {'symbol': 'circle' , 'color': 'indigo', 'size':10},  {'symbol': '-open', 'size':5}
    default_size_legend_points, default_size_legend_round_value, marker_shadow, marker_shadow_offset_x, marker_shadow_offset_y, marker_shadow_add_size = 4, 5, True, 0.00125, 0.003125, 2
    minimum_marker_px, maximum_marker_px, legend_marker_min_px, legend_marker_max_px = 5, 40, 5, 15 # there is a maximum in plotly legends so need to keep legend_marker_max_px small
    marker_symbols = ['circle', 'square', 'diamond', 'x', 'triangle-up', 'pentagon', 'star', 'cross', 'triangle-down',
                      'hexagon', 'hourglass', 'triangle-se', 'triangle-sw', 'star-triangle-down', 'square-x', 'hexagram']
    marker_dict = dict(line=dict(width=1, color='DarkSlateGrey'), opacity=1.0)
    custom_data, hover_lines, added_labels = [], [], set()

    # Remove any lines that are missing row/column; they will end up plotted as empty points; note that this filters/slices df in place and becomes a view (SLICE) or the original dataframe, so we would get a warning
    # when trying to perform actions on the df that you're manipulating a view; adding .copy() ensures we are then working with a new independant dataframe (which is what we want)
    df = df[df[y_variable].notna() & df[x_variable].notna() & (df[y_variable].astype(str).str.strip() != '') & (df[x_variable].astype(str).str.strip() != '')].copy()

    if y_variable.lower() == 'row': # convert rows to alphanumeric characters for the hover template
        if x_variable.lower() == 'column':
            # df.apply applies a function across the dataframe, lambda creates an unnamed function (here taking a single argument 'row' and axis=1 means it's applied row by row, axis=0 would be column by column
            df['well_label'] = df.apply(lambda row: f"{chr(64 + int(row[y_variable]))}{int(row[x_variable])}", axis=1)
            add_hover_and_custom_data(df, custom_data, hover_lines, 'insert', 'well_label', False, 0, None, added_labels=added_labels)
        else:
            # here we select a single column in the dataframe and it's applied element-by-element (one-dimensional series)
            df['row_letter'] = df[y_variable].apply(lambda r: chr(64 + int(r)))
            add_hover_and_custom_data(df, custom_data, hover_lines, 'insert', 'row_letter', False, 0, None, y_variable, added_labels)
            hover_lines.insert(1, f"{x_variable}: %{{x}}")
    else: #  just put x, y
        hover_lines.extend([f"{y_variable}: %{{y}}", f"{x_variable}: %{{x}}"])
        added_labels.add(x_variable)
        if (y_variable.lower() != x_variable.lower()):
            added_labels.add(y_variable)

    # First we normalize the bubble sizes; size-variable should generally be numeric
    if size_variable not in [None, 'None']:
        if is_numeric_dtype(df[size_variable]):
            df[size_variable] = df[size_variable].fillna(0)  # currently we convert those rows to 0 from NaN
            size_variable_column = size_variable
        else:
            size_variable_column = size_variable + category_suffix
        size_norm = (df[size_variable_column] - df[size_variable_column].min()) / (df[size_variable_column].max() - df[size_variable_column].min())
        df['marker_size_px'] = size_norm * (maximum_marker_px - minimum_marker_px) + minimum_marker_px
        marker_dict['size'] = df['marker_size_px']
        add_hover_and_custom_data(df, custom_data, hover_lines, 'append', size_variable, is_numeric_dtype(df[size_variable]), added_labels=added_labels)
    else:
        marker_dict['size'] = default_single_marker['size']

    marker_color_map = None
    if color_variable not in [None, 'None']:
        if is_numeric_dtype(df[color_variable]):  # then we can use the colorscale as usual
            df[color_variable] = df[color_variable].fillna(0)
            marker_dict.update(dict(color=df[color_variable], showscale=True, colorscale=color_scale, colorbar=dict(title=color_variable, orientation='v', x=1.02, y=0.5, len=1.0)))
        else:
            marker_color_map = {
                color: get_discrete_colorscale(color_scale, len(df[color_variable].dropna().unique()))[i % len(get_discrete_colorscale(color_scale, len(df[color_variable].dropna().unique())))]
                for i, color in enumerate(df[color_variable].dropna().unique())
            }
            marker_dict['color'] = df[color_variable].map(marker_color_map)
        add_hover_and_custom_data(df, custom_data, hover_lines, 'append', color_variable, is_numeric_dtype(df[color_variable]), added_labels=added_labels)
    else:
        marker_dict['color'] = default_single_marker['color']

    # Assign symbols to each unique value in the bubble_symbol_variable column
    marker_symbol_map = None
    if symbol_variable not in [None, 'None']:
        if is_numeric_dtype(df[symbol_variable]):
            df[symbol_variable] = df[symbol_variable].fillna(0)
            add_hover_and_custom_data(df, custom_data, hover_lines, 'append', symbol_variable, True, added_labels=added_labels)
        else:
            hover_lines.append(f"{symbol_variable}: %{{text}}")
        marker_symbol_map = {item: marker_symbols[i % len(marker_symbols)] for i, item in enumerate(df[symbol_variable].unique())}
        df['marker_symbol'] = df[symbol_variable].map(marker_symbol_map)
        marker_dict['symbol'] = df['marker_symbol']
        text_data = df[symbol_variable] #if not is_numeric_dtype(df[symbol_variable]) else text_data
    else:
        marker_dict['symbol'] = default_single_marker['symbol']
        text_data = df[x_variable]

    # if size_variable or symbol_variable is numerical and 0 for given data points/rows, modify the marker accordingly
    if size_variable not in [None, 'None'] and is_numeric_dtype(df[size_variable]):
        # create boolean series of len(df) based on condition
        size_zero_mask = df[size_variable] == 0
        df.loc[(size_zero_mask), 'marker_size_px'] = zero_value_marker['size']
        marker_dict['size'] = df['marker_size_px']
    else:
        size_zero_mask = pd.Series(False, index=df.index)
    if symbol_variable not in [None, 'None'] and is_numeric_dtype(df[symbol_variable]):
        symbol_zero_mask = df[symbol_variable] == 0
    else:
        symbol_zero_mask = pd.Series(False, index=df.index)
    zero_mask = size_zero_mask | symbol_zero_mask

    if isinstance(marker_dict['symbol'], pd.Series):
        # first part selects only the rows where zero_mask is True and only the specified column and we reassign to the same column
        df.loc[zero_mask, 'marker_symbol'] = (df.loc[zero_mask, 'marker_symbol'].astype(str) + zero_value_marker['symbol'])
        marker_dict['symbol'] = df['marker_symbol']
    else:
        marker_symbols_array = np.full(len(df), marker_dict['symbol'], dtype=object)
        marker_symbols_array[zero_mask] = [str(sym) + zero_value_marker['symbol'] for sym in marker_symbols_array[zero_mask]]
        marker_dict['symbol'] = marker_symbols_array

    fig = go.Figure()
    x_vals = df[x_variable]
    y_vals = df[y_variable]
    if marker_shadow:
        marker_dict_shadow = marker_dict.copy()
        marker_dict_shadow.update(size=marker_dict['size'] + marker_shadow_add_size, color='rgba(50, 50, 50, 0.35)', opacity=1.0, line=dict(width=0))
        if (isinstance(df[x_variable].dtype, CategoricalDtype)):
            marker_shadow_offset_x = marker_shadow_offset_x*len(df[x_variable].cat.categories)
            x_vals_shadow = [x_val + marker_shadow_offset_x for x_val in df[x_variable + category_suffix]]
            x_vals = df[x_variable + category_suffix]
        else:
            marker_shadow_offset_x = marker_shadow_offset_x*df[x_variable].nunique()
            x_vals_shadow = [x_val + marker_shadow_offset_x for x_val in df[x_variable]]
        if (isinstance(df[y_variable].dtype, CategoricalDtype)):
            marker_shadow_offset_y = marker_shadow_offset_y * len(df[y_variable].cat.categories)
            y_vals_shadow = ([y_val + marker_shadow_offset_y for y_val in df[y_variable + category_suffix]] if (y_variable.lower() == 'row') else [y_val - marker_shadow_offset_y/2 for y_val in df[y_variable + category_suffix]])
            y_vals = df[y_variable + category_suffix]
        else:
            marker_shadow_offset_y = marker_shadow_offset_y * df[y_variable].nunique()
            y_vals_shadow = ([y_val + marker_shadow_offset_y for y_val in df[y_variable]] if (y_variable.lower() == 'row') else [y_val - marker_shadow_offset_y/2 for y_val in df[y_variable]])

        fig.add_trace(go.Scatter(
            x=x_vals_shadow,
            y=y_vals_shadow,
            mode='markers',
            marker=marker_dict_shadow,
            name='',
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=marker_dict,
        text=text_data,
        customdata=np.stack(custom_data, axis=-1) if custom_data else None, # combines list of arrays into a single 2D numpy array; axis=-1 means stacking is done column-wise to each row
        hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
        name='',  # hide trace label in legend
        showlegend=False,  # same idea
    ))

    # Add dummy traces for symbol, size, color variables so that we can have legends for them
    if marker_symbol_map:
        for item, symbol in marker_symbol_map.items():
            display_name = f"{item:.1f}" if isinstance(item, (int, float, numbers.Integral)) else str(item)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol=symbol, size=10, color='black'), name=display_name, legendgroup='symbol_legend', showlegend=True))

    size_legend_values = None
    if size_variable not in [None, 'None']: #and is_numeric_dtype(df[size_variable]):
        if is_numeric_dtype(df[size_variable]):
            min_val, max_val = df[size_variable].min(), df[size_variable].max()
            legend_min = max(default_size_legend_round_value, math.floor(min_val / default_size_legend_round_value) * default_size_legend_round_value)
            legend_max = math.ceil(max_val / default_size_legend_round_value) * default_size_legend_round_value
            vals = np.linspace(legend_min, legend_max, default_size_legend_points)
            vals = [round(val/default_size_legend_round_value)*default_size_legend_round_value for val in vals]
            size_legend_values = [
                (
                    val,
                    ((val - legend_min) / (legend_max - legend_min) if legend_max > legend_min else 0) * (legend_marker_max_px - legend_marker_min_px) + legend_marker_min_px
                )
                for val in vals
            ]
        else:
            category_sizes = df[[size_variable, 'marker_size_px']].drop_duplicates().sort_values('marker_size_px')
            category_sizes['legend_marker_size_px'] = (
                (category_sizes['marker_size_px'] - category_sizes['marker_size_px'].min()) / (category_sizes['marker_size_px'].max() - category_sizes['marker_size_px'].min())
                * (legend_marker_max_px - legend_marker_min_px) + legend_marker_min_px
                if category_sizes['marker_size_px'].max() > category_sizes['marker_size_px'].min() else legend_marker_min_px
            )
            size_legend_values = list(zip(category_sizes[size_variable], category_sizes['legend_marker_size_px'])) # creates a list of tuples from the two columns in cat_sizes
        for val, px in size_legend_values:
            val_str = f"{val:.1f}" if isinstance(val, (int, float, numbers.Integral)) else str(val)
            display_name = f"{size_variable} = {val_str}" if size_variable else val_str
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=px, color='lightgray', opacity=0.5, line=dict(width=0.5, color='gray')), name=display_name, legendgroup='size_legend', showlegend=True))

    if marker_color_map:
        for item, color in marker_color_map.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=str(item), legendgroup='color_legend', showlegend=True))

    # Final layout adjustments
    axis_tick_style = dict(dtick=1, tickfont=dict(color="black", family="Times New Roman", size=16, weight="bold"), ticks='outside', tickcolor='white', ticklen=10)
    fig.update_xaxes(**axis_tick_style) # ** is used to unpack a dictionary into keyword arguments
    # Convert numeric rows to letter ticks if y_variable is 'row'
    if y_variable.lower() == 'row':
        y_vals = sorted(df[y_variable].dropna().unique())
        y_ticks = [chr(64 + int(y)) if isinstance(y, (int, float, numbers.Integral)) and float(y).is_integer() else str(y) for y in y_vals]
        fig.update_yaxes(autorange='reversed', tickvals=y_vals, ticktext=y_ticks, **axis_tick_style)
    else:
        fig.update_yaxes(**axis_tick_style)

    if is_numeric_dtype(df[x_variable]) and x_variable.lower() not in ('column', 'row'):
        fig.update_xaxes(dtick=10)
    if is_numeric_dtype(df[y_variable]) and y_variable.lower() not in ('column', 'row'):
        fig.update_yaxes(dtick=10)

    if marker_shadow:
        if (isinstance(df[x_variable].dtype, CategoricalDtype)):
            fig.update_xaxes(tickvals=list(range(1, len(df[x_variable].cat.categories) + 1)), ticktext=list(df[x_variable].cat.categories))
        if (isinstance(df[y_variable].dtype, CategoricalDtype)):
            fig.update_yaxes(tickvals=list(range(1, len(df[y_variable].cat.categories) + 1)), ticktext=list(df[y_variable].cat.categories))

    top_margin = 140 if graph_title not in [None, ""] else 60
    fig.update_layout(
        margin=dict(t=top_margin, l=10, r=10, b=10),
        title=dict(text=graph_title, font=dict(color='black', family='Times New Roman', size=26, weight='bold'), x=0, xanchor='left', y=0.97, yanchor='top'),
        xaxis=dict(title=x_variable, title_font=dict(size=20, family='Times New Roman', color='darkblue', weight='bold'), title_standoff=30),
        yaxis=dict(title=y_variable, title_font=dict(size=20, family='Times New Roman', color='darkblue', weight='bold'), title_standoff=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='right', x=1.0, font=dict(size=16, family='Times New Roman'), itemsizing='trace'),
        template='plotly_white',
        hoverlabel=dict(bgcolor="black", font_size=14, font_family='Arial', font_color='white', bordercolor='darkgray'),
        shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1, line=dict(color="black", width=1),layer="above")]
    )

    return fig

def generate_heatmap_graph(df, row_variable, column_variable, color_variable, additional_row_variable, additional_column_variable, z_smooth_option, color_scale, graph_title):

    # Remove any lines that are missing row/column, clean up additional_row/column_variables;
    df = df[df[row_variable].notna() & df[column_variable].notna() & (df[row_variable].astype(str).str.strip() != '') & (df[column_variable].astype(str).str.strip() != '')]
    additional_row_variable = additional_row_variable if isinstance(additional_row_variable, list) else ([] if additional_row_variable in [None, 'None'] else [additional_row_variable])
    additional_column_variable = additional_column_variable if isinstance(additional_column_variable, list) else ([] if additional_column_variable in [None, 'None'] else [additional_column_variable])

    z_hover_labels = df.pivot(index=row_variable, columns=column_variable, values=color_variable)
    z_tickvals, z_ticktext = (None, None)
    if (isinstance(df[color_variable].dtype, CategoricalDtype)):
        z_ticktext = list(df[color_variable].cat.categories)
        z_tickvals = list(range(1, len(z_ticktext) + 1))
        z_categorical = df.pivot_table(index=row_variable, columns=column_variable, values=color_variable, aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        category_to_number = {cat: i + 1 for i, cat in enumerate(list(df[color_variable].cat.categories))}
        heatmap_data = z_categorical.map(lambda val: category_to_number.get(val, np.nan))
        color_scale = get_discrete_colorscale(color_scale, len(list(df[color_variable].cat.categories)))
        show_colorbar = False
    else:
        heatmap_data = df.pivot_table(index=row_variable, columns=column_variable, values=color_variable, aggfunc='mean')  # .sort_index().sort_index(axis=1)
        show_colorbar = True

    def format_val(val, is_z=True):
        if pd.isna(val): return "N/A"
        return f"{val:.1f}" if isinstance(val, (int, float, numbers.Integral)) else str(val)

    # Build dictionaries for additional variables (used in additional axis labels and hover labels);
    additional_column_variable_dict = {}
    for col in heatmap_data.columns:
        additional_column_variable_dict[col] = {}
        for var in additional_column_variable:
            item = df.groupby(column_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(col, 'Unknown')
            additional_column_variable_dict[col][var] = str(item)

    additional_row_variable_dict = {}
    for row in heatmap_data.index:
        additional_row_variable_dict[row] = {}
        for var in additional_row_variable:
            item = df.groupby(row_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(row, 'Unknown')
            additional_row_variable_dict[row][var] = str(item)

    custom_hover_data = np.array([
        [[f"{chr(64 + int(y))}{f'{x:.0f}' if isinstance(x, (int, float, numbers.Integral)) and not pd.isna(x) else x}",
          format_val(z_hover_labels.at[y, x] if isinstance(df[color_variable].dtype, CategoricalDtype) else heatmap_data.at[y, x], is_z=True),]
          + [additional_row_variable_dict.get(y, {}).get(var, "") for var in additional_row_variable if var != color_variable and var not in additional_column_variable] # we don't want repeat data in the hover labels
          + [additional_column_variable_dict.get(x, {}).get(var, "") for var in additional_column_variable if var != color_variable and var not in additional_row_variable]
          for x in heatmap_data.columns
        ]
        for y in heatmap_data.index
    ])

    # Build hovertemplate string dynamically;
    hovertemplate = "%{customdata[0]}<br>" + f"{color_variable}: %{{customdata[1]}}<br>"
    if additional_row_variable:
        #  for i, var in enumerate(additional_row_variable):
        for i, var in enumerate([val for val in additional_row_variable if val != color_variable and val not in additional_column_variable]):
            #if var != color_variable and var not in additional_column_variable:
                hovertemplate += f"{var}: %{{customdata[{2 + i}]}}<br>"
    if additional_column_variable:
        start_idx = 2 + len([val for val in additional_row_variable if val != color_variable and val not in additional_column_variable])
        #for j, var in enumerate(additional_column_variable):
        for j, var in enumerate([val for val in additional_column_variable if val != color_variable and val not in additional_row_variable]):
            #if var != color_variable and var not in additional_row_variable:
                hovertemplate += f"{var}: %{{customdata[{start_idx + j}]}}<br>"
    hovertemplate += "<extra></extra>"

    colorbar_dict = dict(title=color_variable)
    if z_tickvals and z_ticktext:
        colorbar_dict.update(tickvals=z_tickvals, ticktext=z_ticktext)
    z_smooth_option = False if z_smooth_option == 'False' else z_smooth_option

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        customdata = custom_hover_data,
        colorscale=color_scale,
        showscale=show_colorbar,
        colorbar=colorbar_dict if show_colorbar else None,
        zsmooth=z_smooth_option,
        hovertemplate=hovertemplate,
    ))

    # Add a color legend if color_variable is categorical
    if not show_colorbar:
        for category, color in zip(df[color_variable].cat.categories, color_scale):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), legendgroup='color_categories', showlegend=True, name=str(category)))

    # Add axis labels on x-axis, including for additional_column_variable if applicable (columns)
    for x_val in heatmap_data.columns:
        if additional_column_variable:
            label_text = f"{f'{x_val:.0f}' if isinstance(x_val, (int, float, numbers.Integral)) and not pd.isna(x_val) else x_val}<br>" + \
                         "<br>".join([additional_column_variable_dict.get(x_val, {}).get(var, 'Unknown') for var in additional_column_variable])
        else:
            label_text = f"{f'{x_val:.0f}' if isinstance(x_val, (int, float, numbers.Integral)) and not pd.isna(x_val) else x_val}"
        fig.add_annotation(x=x_val, y=-0.01, xref="x", yref="paper", text=label_text, showarrow=False, font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='center',
                           yanchor='top', align='center')

    # Add axis labels on y-axis, inluding for additional_row_variable if applicable (rows)
    for y_val in heatmap_data.index:
        alpha_label = chr(64 + int(y_val)) if pd.notna(y_val) >= 1 else "?"
        if additional_row_variable:
            row_meta = "<br>".join([additional_row_variable_dict.get(y_val, {}).get(var, 'Unknown') for var in additional_row_variable])
            label_text = f"{alpha_label}<br>{row_meta}"
        else:
            label_text = alpha_label
        fig.add_annotation(x=-0.005, y=y_val, xref="x domain", yref="y", text=label_text, showarrow=False, font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='right',
                           yanchor='middle', align='center')

    fig.update_xaxes(tickmode='array', tickvals=list(heatmap_data.columns), showticklabels=False, ticks='outside', tickcolor='white', ticklen=10)
    fig.update_yaxes(tickmode='array', tickvals=list(heatmap_data.index), autorange='reversed', showticklabels=False, ticks='outside', tickcolor='white', ticklen=10)

    top_margin = 140 if graph_title not in [None, ""] else 60 # was previously 110
    fig.update_layout(
        margin=dict(t=top_margin, b=90, l=130),
        title=dict(text=graph_title, font=dict(color="black", family="Times New Roman", size=26, weight="bold"), x=0, xanchor='left', y=0.97, yanchor='top'),
        xaxis_title=column_variable, xaxis_title_font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        yaxis_title=row_variable, yaxis_title_font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        xaxis=dict(title_standoff=60),
        yaxis=dict(title_standoff=60),
        legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.02, font=dict(size=16, family='Times New Roman'), itemsizing='trace')
    )

    return fig

def generate_piecharts_graph(df, row_variable, column_variable, pie_charts_variables, additional_row_variable, additional_column_variable, normalization_method, normalization_value, colorscale, graph_title):

    def compute_pie_values_and_labels(values, labels, normalization_method, normalization_value, well_id):
        pie_border = dict(color='black', width=1)
        values = [float(val) if not pd.isna(val) else 0 for val in values]
        all_zero = 0

        if int(all(v == 0 for v in values)):
            pie_vals = [1]
            labels = ['Empty']
            hover_info = [f"{well_id}<br>Empty: 0%"]
            all_zero = 1
        elif normalization_method == 0:
            total = sum(values) or 1
            pie_vals = [(v / total) * 100 for v in values]
            # changed to values from original_values and updated_labels to labels
            hover_info = [f"{well_id}<br>{label}: {val:.1f}% [{orig:.3g}]" for label, val, orig in
                          zip(labels, pie_vals, values)]
        elif normalization_method == 1:
            pie_vals = values.copy()
            total = sum(pie_vals)
            hover_info = [f"{well_id}<br>{label}: {val:.1f}%" for label, val in zip(labels, pie_vals)]
            if total < 100:
                unknown_percent = 100 - total
                labels.append("Unknown")  # changed
                pie_vals.append(unknown_percent)
                hover_info.append(f"{well_id}<br>Unknown: {unknown_percent:.1f}%")
            elif total > 100:
                pie_border = dict(color='red', width=2)
        elif normalization_method == 2:
            max_val = normalization_value or 1
            pie_vals = [(v / max_val) * 100 for v in values]
            hover_info = [f"{well_id}<br>{label}: {val:.1f}% [{(val * max_val / 100):.3g}]" for label, val in
                          zip(labels, pie_vals)]
            total = sum(pie_vals)
            if total < 100:
                unknown_percent = 100 - total
                unknown_value = max_val * (unknown_percent / 100)
                labels.append("Unknown")
                pie_vals.append(unknown_percent)
                hover_info.append(f"{well_id}<br>Unknown: {unknown_percent:.1f}% [{unknown_value:.3g}]")
            elif total > 100:
                pie_border = dict(color='red', width=2)
        else:
            pie_vals = [1]
            labels = ['Invalid normalization method']
            hover_info = [f"{well_id}<br>Error: invalid normalization method"]
        return pie_vals, labels, hover_info, pie_border, all_zero

    # Set some defaults;
    selected_slice_colors_from_colorscale = get_discrete_colorscale(colorscale, len(pie_charts_variables))
    df[row_variable], df[column_variable] = pd.to_numeric(df[row_variable], errors='coerce'), pd.to_numeric(df[column_variable], errors='coerce')
    base_unit, spacing_px, plate_rows, plate_columns = 100, 10, list(range(1, int(df[row_variable].max()) + 1)), list(range(1, int(df[column_variable].max()) + 1))
    plot_width, plot_height = base_unit*(len(plate_columns)) + ((len(plate_columns))-1)*spacing_px, base_unit*(len(plate_rows)) + ((len(plate_rows))-1)*spacing_px # move down
    spacing_x, spacing_y, cell_width, cell_height = spacing_px/plot_width,  spacing_px/plot_height, 1/(len(plate_columns)), 1/(len(plate_rows))
    pie_width, pie_height = cell_width - spacing_x, cell_height - spacing_y
    added_labels = set()

    # Build maps for the additional row/column variables
    additional_column_variable_hover_map, additional_column_variable_axis_map = {}, {}
    if (additional_column_variable not in [None, 'None']):
        if not isinstance(additional_column_variable, list):
            additional_column_variable = [additional_column_variable]
        for col in plate_columns:
            hover_values = []
            axis_values = []
            for var in additional_column_variable:
                item = df.groupby(column_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(col, 'Unknown')
                axis_values.append(str(item))
                if not any(var in hv for hv in hover_values):
                    hover_values.append(f"{var}: {item}")
            additional_column_variable_hover_map[col] = "<br>".join(hover_values)
            additional_column_variable_axis_map[col] = "<br>".join(axis_values)

    additional_row_variable_hover_map, additional_row_variable_axis_map = {}, {}
    if (additional_row_variable not in [None, 'None']):
        if not isinstance(additional_row_variable, list):
            additional_row_variable = [additional_row_variable]
        for row in plate_rows:
            hover_values = []
            axis_values = []
            for var in additional_row_variable:
                item = df.groupby(row_variable, observed=False)[var].agg(lambda x: x.unique()[0]).to_dict().get(row, 'Unknown')
                axis_values.append(str(item))
                if not any(var in hv for hv in hover_values):
                    hover_values.append(f"{var}: {item}")
            additional_row_variable_hover_map[row] = "<br>".join(hover_values)
            additional_row_variable_axis_map[row] = "<br>".join(axis_values)
    # Generate pie traces manually
    fig = go.Figure()
    for row_index, row in enumerate(plate_rows):  # this iterates through each element of the array, providing the 0-based index and the value
        for col_index, col in enumerate(plate_columns):
            row_data = df[(df[row_variable] == row) & (df[column_variable] == col)]
            x_center = (col_index + 0.5) * cell_width
            y_center = 1 - (row_index + 0.5) * cell_height
            pie_domain = dict(x=[x_center - pie_width / 2, x_center + pie_width / 2], y=[y_center - pie_height / 2, y_center + pie_height / 2])
            # Dealing with additional variables for hover_labels
            row_meta = additional_row_variable_hover_map.get(row, '') if additional_row_variable else ''
            col_meta = additional_column_variable_hover_map.get(col, '') if additional_column_variable else ''
            well_label = f"{chr(64 + row)}{col}"
            if not row_data.empty:
                row_values = [row_data.iloc[0][var] for var in pie_charts_variables]
                updated_vals, updated_labels, hover_text, updated_pie_border, all_zero = compute_pie_values_and_labels(row_values, pie_charts_variables.copy(), normalization_method, normalization_value, \
                    well_label)
                # Adding additional row/column variable info after
                if col_meta or row_meta:
                    meta_text = f"{row_meta}{'<br>' if row_meta and col_meta else ''}{col_meta}"
                    hover_text = [f"{ht}<br>{meta_text}" for ht in hover_text]
                updated_colors = selected_slice_colors_from_colorscale + ['white'] * (len(updated_labels) - len(selected_slice_colors_from_colorscale)) if all_zero==0 else ['white']
                pie_args = {'values': updated_vals, 'labels': updated_labels, 'marker': dict(colors=updated_colors, line=updated_pie_border), 'domain': pie_domain, 'text': hover_text,
                            'hoverinfo': 'text', 'textinfo': 'none', 'sort': False}
            else: # missing well: white pie with black border
                pie_args = {'values': [1], 'labels': ['Missing'], 'marker': dict(colors=['white'], line=dict(color='black', width=1)), 'domain': pie_domain, 'hoverinfo': 'skip', 'textinfo': 'none'}
            fig.add_trace(go.Pie(**pie_args))

    # Add axis labels on x-axis (columns);
    fig.add_annotation(x=0.5, y=-0.08, text=column_variable, showarrow=False, xref="paper", yref="paper", font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"), xanchor='center', yanchor='top')
    for col_index, col in enumerate(plate_columns):
        x = (col_index + 0.5) * cell_width
        label_text = f"{col}<br>{additional_column_variable_axis_map.get(col, 'Unknown')}" if additional_column_variable else str(col)
        fig.add_annotation(x=x, y=-0.01, text=label_text, showarrow=False, xref="paper", yref="paper", font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='center', yanchor='top')

    # Add axis labels on y-axis (rows);
    fig.add_annotation(x=-0.08, y=0.5, text=row_variable, showarrow=False, xref="paper", yref="paper", font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"), xanchor='right', yanchor='middle')
    for row_index, row in enumerate(plate_rows):
        y = 1 - (row_index + 0.5) * cell_height
        label_text = chr(64 + int(row))
        if additional_row_variable:
            label_text += f"<br>{additional_row_variable_axis_map.get(row, 'Unknown')}"
        fig.add_annotation(x=-0.01, y=y, text=label_text, showarrow=False, xref="paper", yref="paper", font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='right', yanchor='middle')

    # Update layout
    top_margin = 140 if graph_title not in [None, ""] else 60  # was previously 110
    fig.update_layout(
        title=dict(text=graph_title, font=dict(color="black", family="Times New Roman", size=26, weight="bold"), x=0, xanchor='left', y=0.97, yanchor='top'),
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', font=dict(size=16), itemsizing='constant', tracegroupgap=5, yanchor='top', bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0)'),
        margin=dict(t=top_margin, b=90, l=130),
        # add rectangle around pie charts to create plate layout
        shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1, line=dict(color='black', width=2), fillcolor='rgba(0,0,0,0)')]
    )

    return fig

