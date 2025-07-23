import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import math
import numpy as np
from miscFunctions import convert_categorical, get_discrete_colorscale


def generate_parallel_coordinates_graph(df, parallel_variables, color_variable, color_scale, graph_title,
                                        category_suffix):
    if not parallel_variables:
        print("Some variables must be selected for a parallel coordinates graph.")
        return
    df[color_variable] = df[color_variable].fillna(
        0)  # currently we convert those rows to 0 from NaN, but do we want to drop them instead?
    plot_dimensions = []

    for col in parallel_variables:

        if isinstance(df[col].dtype, CategoricalDtype):

            categories = df[col].cat.categories
            num_categories = len(categories)
            tickvals = list(range(1, num_categories + 1))
            ticktext = list(categories)
            dim = dict(
                range=[1, num_categories],
                tickvals=tickvals,
                ticktext=ticktext,
                label=col.title(),
                values=df[col + category_suffix]
            )
            plot_dimensions.append(dim)
        elif is_numeric_dtype(df[col]) and not col.endswith(category_suffix):

            plot_dimensions.append(dict(
                # range=[math.floor(df[col].min()/5)*5,math.ceil(df[col].max()/10)*10],
                range=[df[col].min(), df[col].max()],
                label=col.title(),
                values=df[col]))

        line_dict = {}
        if (color_variable not in ['None', None]):
            line_dict = dict(
                color=df[color_variable],
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title=color_variable
                ),
                cmin=0,
                # cmin = df[parallel_color_variable].min(),
                cmax=math.ceil(df[color_variable].max() / 10) * 10  # + 10
                # cmax=100,
                # cmax = df[parallel_color_variable].max() + 20
            )

        fig = go.Figure(data=go.Parcoords(
            line=line_dict,
            dimensions=plot_dimensions,
            unselected=dict(line=dict(color='lightgray', opacity=0.75)),
            labelfont=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
            tickfont=dict(color="black", family="Times New Roman", size=16, weight="bold"),
        ))
        fig.update_layout(
            title=dict(text=graph_title, font=dict(color="black", family="Times New Roman", size=26, weight="bold"),
                       x=0, xanchor='left', y=0.97, yanchor='top'),
            margin=(dict(t=140))  # this controls tha padding around the entire graph area, including the title and axes
        )

    # fig.show(renderer='browser')
    return fig


def generate_scatter_bubble_graph(df, x_variable, y_variable, size_variable, symbol_variable, color_variable,
                                  color_scale, graph_title, category_suffix):
    if (x_variable is None) or (y_variable is None):
        print("X and Y variables must both be assigned values.")
        return

    # Set some defaults
    default_single_marker_symbol, default_single_marker_color, default_single_marker_size, = 'circle', 'red', 10
    default_size_legend_points, default_size_legend_round_value = 4, 5
    minimum_bubble_px, maximum_bubble_px = 10, 40
    marker_dict = dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.8)
    customdata_cols = []
    hover_template_lines = []

    df_clean = df.copy()
    df_clean = df_clean[
        df_clean[y_variable].notna() & df_clean[x_variable].notna() &
        (df_clean[y_variable].astype(str).str.strip() != '') &
        (df_clean[x_variable].astype(str).str.strip() != '')
        ]

    if y_variable.lower() == 'row':
        if x_variable.lower() == 'column':
            # df['well_label'] = df.apply(lambda row: f"{chr(64 + int(row[next(col for col in df.columns if col.lower() == 'row')]))}{int(row[next(col for col in df.columns if col.lower() == 'column')])}", axis=1)
            df_clean['well_label'] = df_clean.apply(
                lambda
                    row: f"{chr(64 + int(row[next(col for col in df_clean.columns if col.lower() == 'row')]))}{int(row[next(col for col in df_clean.columns if col.lower() == 'column')])}"
                if pd.notna(row[next(col for col in df_clean.columns if col.lower() == 'row')]) and pd.notna(
                    row[next(col for col in df_clean.columns if col.lower() == 'column')])
                else None,
                axis=1
            )
            customdata_cols.insert(0, df_clean['well_label'])  # Make sure it's at the front of customdata
            hover_template_lines.insert(0, "Well: %{customdata[0]}")
        else:
            df_clean['row_letter'] = df_clean[next(col for col in df_clean.columns if col.lower() == 'row')].apply(
                lambda r: chr(64 + int(r)) if pd.notna(r) else None
            )
            customdata_cols.insert(0, df_clean['row_letter'])  # First column = row letter
            hover_template_lines.insert(0, f"{y_variable}: %{{customdata[0]}}")
            hover_template_lines.insert(1, f"{x_variable}: %{{x}}")
    else:
        hover_template_lines.insert(0, f"{y_variable}: %{{y}}")
        hover_template_lines.insert(1, f"{x_variable}: %{{x}}")

    # First we normalize the bubble sizes; size-variable should generally be numeric
    if size_variable not in [None, 'None']:
        customdata_cols.append(df_clean[size_variable])

        if not is_numeric_dtype(df_clean[size_variable]):
            if df_clean[size_variable].isna().any():
                if 'Missing' not in df_clean[size_variable].cat.categories:
                    df_clean[size_variable] = df_clean[size_variable].cat.add_categories(['Missing'])
                df_clean[size_variable] = df_clean[size_variable].fillna('Missing')
            convert_categorical(df_clean, size_variable, category_suffix)
            size_norm = (df_clean[size_variable + category_suffix] - df_clean[
                size_variable + category_suffix].min()) / (df_clean[size_variable + category_suffix].max() - df_clean[
                size_variable + category_suffix].min())
            hover_template_lines.append(f"{size_variable}: %{{customdata[{len(customdata_cols) - 1}]}}")
        else:
            df_clean[size_variable] = df_clean[size_variable].fillna(0)  # currently we convert those rows to 0 from NaN
            size_norm = (df_clean[size_variable] - df_clean[size_variable].min()) / (
                    df_clean[size_variable].max() - df_clean[size_variable].min())
            hover_template_lines.append(f"{size_variable}: %{{customdata[{len(customdata_cols) - 1}]:.2f}}")
        df_clean['bubble_size_px'] = size_norm * (maximum_bubble_px - minimum_bubble_px) + minimum_bubble_px
        marker_dict['size'] = df_clean['bubble_size_px']
    else:
        marker_dict['size'] = default_single_marker_size

    if color_variable not in [None, 'None']:
        if is_numeric_dtype(df_clean[color_variable]):  # then we can use the colorscale as usual
            bubble_color_array = df_clean[color_variable]
            bubble_color_scale_updated = color_scale
            bubble_colorbar_dict = dict(title=color_variable, orientation='v', x=1.02, xanchor='left', y=0.5,
                                        yanchor='middle', len=1.0)

            customdata_cols.append(df_clean[color_variable])
            hover_template_lines.append(f"{color_variable}: %{{customdata[{len(customdata_cols) - 1}]:.2f}}")
        else:  # is categorical, so no colorscale and take colors from specified color scale
            if df_clean[color_variable].isna().any():
                if 'Missing' not in df_clean[color_variable].cat.categories:
                    df_clean[color_variable] = df_clean[color_variable].cat.add_categories(['Missing'])
                df_clean[color_variable] = df_clean[color_variable].fillna('Missing')

            bubble_color_variable_unique_values = df_clean[color_variable].dropna().unique()

            selected_colors_from_color_scale = get_discrete_colorscale(color_scale,
                                                                       len(bubble_color_variable_unique_values))

            bubble_color_map = {
                color: selected_colors_from_color_scale[i % len(selected_colors_from_color_scale)]
                for i, color in enumerate(bubble_color_variable_unique_values)
            }
            df_clean[f'{color_variable}' + "_color"] = df_clean[color_variable].map(bubble_color_map)
            bubble_color_array = df_clean[f'{color_variable}' + "_color"]
            bubble_color_scale_updated = None
            bubble_colorbar_dict = None
            customdata_cols.append(df_clean[color_variable])
            hover_template_lines.append(f"{color_variable}: %{{customdata[{len(customdata_cols) - 1}]}}")
        marker_dict['color'] = bubble_color_array
        marker_dict['showscale'] = (is_numeric_dtype(df_clean[color_variable]))
        marker_dict['colorscale'] = bubble_color_scale_updated
        marker_dict['colorbar'] = bubble_colorbar_dict
    else:
        marker_dict['color'] = default_single_marker_color

    # Assign symbols to each unique value in the bubble_symbol_variable column
    if symbol_variable not in [None, 'None']:
        bubble_symbols = ['circle', 'square', 'diamond', 'x', 'triangle-up', 'pentagon', 'star',
                          'cross']  # up to 8 currently defined
        if not is_numeric_dtype(df_clean[symbol_variable]):
            if df_clean[symbol_variable].isna().any():
                if 'Missing' not in df_clean[symbol_variable].cat.categories:
                    df_clean[symbol_variable] = df_clean[symbol_variable].cat.add_categories(['Missing'])
                df_clean[symbol_variable] = df_clean[symbol_variable].fillna('Missing')
        else:
            df_clean[symbol_variable] = df_clean[symbol_variable].fillna(
                0)  # currently we convert those rows to 0 from NaN

        bubble_symbol_variable_symbols = {item: bubble_symbols[i % len(bubble_symbols)] for i, item in
                                          enumerate(df_clean[symbol_variable].unique())}
        df_clean[f'{symbol_variable}' + "_symbol"] = df_clean[symbol_variable].map(bubble_symbol_variable_symbols)
        marker_dict['symbol'] = df_clean[f'{symbol_variable}_symbol']
        text_data = df_clean[symbol_variable]
        if is_numeric_dtype(df_clean[symbol_variable]):
            customdata_cols.append(df_clean[symbol_variable])
            hover_template_lines.append(f"{symbol_variable}: %{{customdata[{len(customdata_cols) - 1}]:.2f}}")
        else:
            hover_template_lines.append(f"{symbol_variable}: %{{text}}")
    else:
        marker_dict['symbol'] = default_single_marker_symbol
        text_data = df_clean[x_variable]

    # Combine customdata if it exists;
    customdata_array = np.stack(customdata_cols, axis=-1) if customdata_cols else None
    # Create the main bubble scatter trace;
    fig = go.Figure(go.Scatter(
        x=df_clean[x_variable],
        y=df_clean[y_variable],
        mode='markers',
        marker=marker_dict,
        text=text_data,
        hovertemplate="<br>".join(hover_template_lines) + "<extra></extra>",
        customdata=customdata_array,
        name='',  # hide trace label in legend
        showlegend=False,  # same idea
        # labelfont = dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        # tickfont = dict(color="black", family="Times New Roman", size=16, weight="bold"),
    ))

    # Add dummy traces for symbol, size, color variables so that we can have legends for them
    # if symbol_variable not in [None, 'None']:
    #   for item, symbol in bubble_symbol_variable_symbols.items():
    #       fig.add_trace(go.Scatter(
    #           x=[None], y=[None], mode='markers', marker=dict(size=10, symbol=symbol, color='gray'), name=item,
    #           legendgroup='symbol_legend', showlegend=True)
    #       )

    if symbol_variable not in [None, 'None']:
        for item, symbol in bubble_symbol_variable_symbols.items():
            # Format item: round if numeric, leave as-is if not
            if isinstance(item, (int, float)):
                display_name = f"{item:.1f}"
            else:
                display_name = str(item)

            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, symbol=symbol, color='gray'),
                name=display_name,
                legendgroup='symbol_legend',
                showlegend=True
            ))

    if size_variable not in [None, 'None']:
        if not is_numeric_dtype(df_clean[size_variable]):
            size_legend_df_clean = df_clean[[size_variable, 'bubble_size_px']].drop_duplicates().sort_values(
                'bubble_size_px')
            for _, row in size_legend_df_clean.iterrows():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=row['bubble_size_px'], color='lightgray', opacity=0.5,
                                line=dict(width=0.5, color='gray')), name=str(row[size_variable]),
                    legendgroup="size_legend", showlegend=True)
                )
        else:
            min_val = df_clean[size_variable].min()
            max_val = df_clean[size_variable].max()
            legend_min = max(default_size_legend_round_value, math.floor(min_val / default_size_legend_round_value) * 5)
            legend_max = math.ceil(max_val / default_size_legend_round_value) * default_size_legend_round_value
            legend_values = np.linspace(legend_min, legend_max, default_size_legend_points)
            for val in legend_values:
                # Normalize using original data's min/max (not the padded range)
                size_norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
                size_px = size_norm * (maximum_bubble_px - minimum_bubble_px) + minimum_bubble_px
                rounded_val = round(val / default_size_legend_round_value) * default_size_legend_round_value
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers',
                    marker=dict(size=size_px, color='lightgray', opacity=0.5, line=dict(width=0.5, color='gray')),
                    name=f"{size_variable} = {rounded_val:.1f}", legendgroup="size_legend", showlegend=True)
                )
    if color_variable not in [None, 'None']:
        if (not is_numeric_dtype(df_clean[color_variable])):
            for item, color in bubble_color_map.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='markers', marker=dict(size=10, color=color), name=item,
                    legendgroup='color_legend', showlegend=True)
                )

    # Final layout adjustments
    fig.update_xaxes(
        # tickmode='linear',
        dtick=1,
        tickfont=dict(color="black", family="Times New Roman", size=16, weight="bold"),
        ticks='outside',
        tickcolor='white',
        ticklen=10
    )

    # Convert numeric rows to letter ticks if y_variable is 'row'
    if y_variable.lower() == 'row':
        y_vals = sorted(df_clean[y_variable].dropna().unique())
        y_ticks = [chr(ord('A') + int(y) - 1) if str(y).isdigit() else str(y) for y in y_vals]
        fig.update_yaxes(
            autorange='reversed',
            dtick=1,
            tickvals=y_vals,
            ticktext=y_ticks,
            tickfont=dict(color="black", family="Times New Roman", size=16, weight="bold"),
            ticks='outside',
            tickcolor='white',
            ticklen=10
        )
    else:
        fig.update_yaxes(
            autorange='reversed',
            dtick=1,
            tickfont=dict(color="black", family="Times New Roman", size=16, weight="bold"),
            ticks='outside',
            tickcolor='white',
            ticklen=10
        )

    fig.update_layout(
        margin=dict(t=140, l=10, r=10, b=10),
        title=dict(
            text=graph_title,
            font=dict(color='black', family='Times New Roman', size=26, weight='bold'),
            x=0,
            xanchor='left',
            y=0.97,
            yanchor='top',
        ),
        # height=1200,
        # width=2000,
        xaxis=dict(
            title=x_variable,
            title_font=dict(size=20, family='Times New Roman', color='darkblue', weight='bold'),
            title_standoff=30,
            # showline=True,
            # linewidth=1,
            # linecolor='black'
        ),
        yaxis=dict(
            title=y_variable,
            title_font=dict(size=20, family='Times New Roman', color='darkblue', weight='bold'),
            title_standoff=30,
            # showline = True,
            # linewidth = 1,
            # linecolor = 'black'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            # y=-0.25,
            # xanchor='center',
            # x=0.5,
            y=-0.4,
            xanchor='right',
            x=1.0,
            font=dict(size=16, family='Times New Roman')
        ),
        # margin=dict(b=200),
        template='plotly_white',
        hoverlabel=dict(
            bgcolor="black",
            font_size=14,
            font_family='Arial',
            font_color='white',
            bordercolor='darkgray'
        ),
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    color="black",
                    width=1
                ),
                layer="above"
            )
        ]
    )

    # fig.show(renderer='browser')
    return fig


def generate_heatmap_graph(df, row_variable, column_variable, color_variable, additional_row_variable,
                           additional_column_variable, z_smooth_option, color_scale, graph_title):
    df_clean = df.copy()
    df_clean = df_clean[
        df_clean[row_variable].notna() & df_clean[column_variable].notna() &
        (df_clean[row_variable].astype(str).str.strip() != '') &
        (df_clean[column_variable].astype(str).str.strip() != '')
        ]
    z_hover_labels = df_clean.pivot(index=row_variable, columns=column_variable, values=color_variable)
    z_tickvals, z_ticktext = (None, None)

    if (isinstance(df[color_variable].dtype, CategoricalDtype)):
        z_ticktext = list(df[color_variable].cat.categories)
        z_tickvals = list(range(1, len(z_ticktext) + 1))
        z_categorical = df_clean.pivot_table(index=row_variable, columns=column_variable, values=color_variable,
                                             aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        category_to_number = {cat: i + 1 for i, cat in enumerate(df[color_variable].cat.categories)}
        heatmap_data = z_categorical.applymap(lambda val: category_to_number.get(val, np.nan))
        color_scale = get_discrete_colorscale(color_scale, len(df[color_variable].cat.categories))
    else:
        heatmap_data = df_clean.pivot_table(index=row_variable, columns=column_variable, values=color_variable,
                                            aggfunc='mean')  # .sort_index().sort_index(axis=1)
        color_scale = color_scale

    def format_val(val, is_z=True):
        if pd.isna(val): return "N/A"
        if isinstance(val, (int, float)):
            return f"{val:.1f}"
        return str(val)
        # if pd.isna(val): return "N/A"
        # return str(val) if is_z else (f"{val:.0f}" if isinstance(val, (int, float)) else str(val))

    # convert row values to alpha characters and column row-column in an alphanumeric character
    # custom_hover_data = np.array([
    #    [[f"{chr(64 + int(y))}{x}", f"{format_val(z_hover_labels.at[y, x] if isinstance(df[color_variable].dtype, CategoricalDtype) else heatmap_data.at[y, x], is_z=True)}"]
    #        for x in heatmap_data.columns]
    #    for y in heatmap_data.index
    # ])
    custom_hover_data = np.array([
        [[f"{chr(64 + int(y))}{f'{x:.0f}' if isinstance(x, (int, float)) and not pd.isna(x) else x}",
          f"{format_val(z_hover_labels.at[y, x] if isinstance(df[color_variable].dtype, CategoricalDtype) else heatmap_data.at[y, x], is_z=True)}"]
         for x in heatmap_data.columns]
        for y in heatmap_data.index
    ])

    colorbar_dict = dict(title=color_variable)
    if z_tickvals and z_ticktext:
        colorbar_dict.update(tickvals=z_tickvals, ticktext=z_ticktext)
    z_smooth_option = False if z_smooth_option == 'False' else z_smooth_option

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        customdata=custom_hover_data,
        colorscale=color_scale,
        colorbar=colorbar_dict,
        zsmooth=z_smooth_option,
        hovertemplate="%{customdata[0]}<br>" + f"{color_variable}: %{{customdata[1]}}<extra></extra>",
    ))

    # Add axis labels/annotations on x-axis (columns)
    if (additional_column_variable is not None):
        additional_column_variable_map = df.groupby(column_variable, observed=False)[additional_column_variable].agg(
            lambda x: x.unique()[0]).to_dict()
    for x_val in heatmap_data.columns:
        # label_text = f"{x_val}<br>{additional_column_variable_map.get(x_val, 'Unknown')}" if additional_column_variable else str(x_val)
        label_text = f"{f'{x_val:.0f}' if isinstance(x_val, (int, float)) and not pd.isna(x_val) else x_val}<br>{additional_column_variable_map.get(x_val, 'Unknown')}" if additional_column_variable else f"{f'{x_val:.0f}' if isinstance(x_val, (int, float)) and not pd.isna(x_val) else x_val}"
        fig.add_annotation(x=x_val, y=-0.01, xref="x", yref="paper", text=label_text, showarrow=False,
                           font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='center',
                           yanchor='top', align='center')

    # Add axis labels on y-axis (rows)
    if (additional_row_variable is not None):
        additional_row_variable_map = df.groupby(row_variable, observed=False)[additional_row_variable].agg(
            lambda x: x.unique()[0]).to_dict()

    for y_val in heatmap_data.index:
        alpha_label = chr(64 + int(y_val)) if pd.notna(y_val) >= 1 else "?"
        row_meta = additional_row_variable_map.get(y_val, 'Unknown') if additional_row_variable else ''

        label_text = f"{alpha_label}<br>{row_meta}" if additional_row_variable else alpha_label
        fig.add_annotation(x=-0.005, y=y_val, xref="x domain", yref="y", text=label_text, showarrow=False,
                           font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='right',
                           yanchor='middle', align='center')

    fig.update_xaxes(
        tickmode='array', tickvals=list(heatmap_data.columns),
        showticklabels=False, ticks='outside', tickcolor='white', ticklen=10
    )

    fig.update_yaxes(
        tickmode='array', tickvals=list(heatmap_data.index), autorange='reversed',
        showticklabels=False, ticks='outside', tickcolor='white', ticklen=10
    )

    fig.update_layout(
        # margin=dict(b=80, l=120),
        margin=dict(t=110, b=90, l=130),
        title=dict(text=graph_title, font=dict(color="black", family="Times New Roman", size=26, weight="bold"), x=0,
                   xanchor='left', y=0.97, yanchor='top'),
        xaxis_title=column_variable,
        xaxis_title_font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        # xaxis_side='top',
        yaxis_title=row_variable,
        # yaxis_title_standoff=50,
        yaxis_title_font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        xaxis=dict(
            title_standoff=60
        ),
        yaxis=dict(
            title_standoff=60
        )
    )

    # fig.show(renderer='browser')
    return fig


def compute_pie_values_and_labels(values, labels, normalization_method, normalization_value, well_id):
    updated_pie_border = dict(color='black', width=1)
    values = [float(v) if not pd.isna(v) else 0 for v in values]
    all_zero = 0

    if all(v == 0 for v in values):
        pie_vals = [1]
        labels = ['Empty']
        hover_info = [f"{well_id}<br>Empty: 0%"]
        all_zero = 1
        # return [1], ['Empty'], [f"{well_id}<br>Empty: 0%"], updated_pie_border
    elif normalization_method == 0:
        total = sum(values) or 1
        pie_vals = [(v / total) * 100 for v in values]
        hover_info = [
            f"{well_id}<br>{label}: {val:.1f}% [{orig:.3g}]"
            for label, val, orig in zip(labels, pie_vals, values)
            # changed to values from original_values and updated_labels to labels
        ]
    elif normalization_method == 1:
        pie_vals = values.copy()
        total = sum(pie_vals)
        hover_info = [
            f"{well_id}<br>{label}: {val:.1f}%"
            for label, val in zip(labels, pie_vals)
        ]
        if total < 100:
            unknown_percent = 100 - total
            labels.append("Unknown")  # changed
            pie_vals.append(unknown_percent)
            hover_info.append(f"{well_id}<br>Unknown: {unknown_percent:.1f}%")
        elif total > 100:
            updated_pie_border = dict(color='red', width=2)
    elif normalization_method == 2:

        max_val = normalization_value or 1

        pie_vals = [(v / max_val) * 100 for v in values]
        hover_info = [
            f"{well_id}<br>{label}: {val:.1f}% [{(val * max_val / 100):.3g}]"
            for label, val in zip(labels, pie_vals)
        ]
        total = sum(pie_vals)
        if total < 100:
            unknown_percent = 100 - total
            unknown_value = max_val * (unknown_percent / 100)
            labels.append("Unknown")
            pie_vals.append(unknown_percent)
            hover_info.append(f"{well_id}<br>Unknown: {unknown_percent:.1f}% [{unknown_value:.3g}]")
        elif total > 100:
            updated_pie_border = dict(color='red', width=2)

    else:
        pie_vals = [1]
        labels = ['Invalid normalization method']
        hover_info = [f"{well_id}<br>Error: invalid normalization method"]
    return pie_vals, labels, hover_info, updated_pie_border, all_zero


def generate_piecharts_graph(df, row_variable, column_variable, pie_charts_variables, additional_row_variable,
                             additional_column_variable, normalization_method, normalization_value, colorscale,
                             graph_title):
    if normalization_method == 2:
        try:
            normalization_value = float(normalization_value)
            if (normalization_value <= 0):
                print("Normalization value cannot be <= 0; exiting.")
                return
        except (ValueError, TypeError):
            print("Normalization value entered cannot be converted to float; exiting.")
            return
    print(f"Method: {normalization_method}, Value: {normalization_value} ({type(normalization_value)})")

    # Set some defaults;
    selected_slice_colors_from_colorscale = get_discrete_colorscale(colorscale, len(pie_charts_variables))
    base_unit, spacing_px = 100, 10

    df[row_variable] = pd.to_numeric(df[row_variable], errors='coerce')
    df[column_variable] = pd.to_numeric(df[column_variable], errors='coerce')

    plate_rows, plate_columns = list(range(1, int(df[row_variable].max()) + 1)), list(
        range(1, int(df[column_variable].max()) + 1))
    plot_width, plot_height = base_unit * (len(plate_columns)) + ((len(plate_columns)) - 1) * spacing_px, base_unit * (
        len(plate_rows)) + ((len(plate_rows)) - 1) * spacing_px  # move down
    spacing_x, spacing_y = spacing_px / plot_width, spacing_px / plot_height
    cell_width, cell_height = 1 / (len(plate_columns)), 1 / (len(plate_rows))
    pie_width, pie_height = cell_width - spacing_x, cell_height - spacing_y

    # Generate pie traces manually
    fig = go.Figure()
    for row_index, row in enumerate(
            plate_rows):  # this iterates through each element of the array, providing the 0-based index and the value
        for col_index, col in enumerate(plate_columns):
            row_data = df[(df[row_variable] == row) & (df[column_variable] == col)]

            x_center = (col_index + 0.5) * cell_width
            y_center = 1 - (row_index + 0.5) * cell_height
            pie_domain = dict(
                x=[x_center - pie_width / 2, x_center + pie_width / 2],
                y=[y_center - pie_height / 2, y_center + pie_height / 2]
            )
            if not row_data.empty:
                row_values = [row_data.iloc[0][var] for var in pie_charts_variables]
                updated_vals, updated_labels, hover_text, updated_pie_border, all_zero = compute_pie_values_and_labels(
                    row_values, pie_charts_variables.copy(), normalization_method, normalization_value,
                    f"{chr(64 + row)}{col}")

                if all_zero == 0:
                    final_colors = selected_slice_colors_from_colorscale + ['white'] * (
                                len(updated_labels) - len(selected_slice_colors_from_colorscale))
                else:
                    final_colors = ['white']
                # pie_args = {'values': updated_vals, 'labels': updated_labels, 'marker': dict(colors=final_colors, line=dict(color='black', width=1)),
                #             'domain': pie_domain, 'text': hover_text, 'hoverinfo': 'text', 'textinfo': 'none',
                #             'sort': False}

                pie_args = {'values': updated_vals, 'labels': updated_labels,
                            'marker': dict(colors=final_colors, line=updated_pie_border),
                            'domain': pie_domain, 'text': hover_text, 'hoverinfo': 'text', 'textinfo': 'none',
                            'sort': False}

            else:  # missing well: white pie with black border
                pie_args = {'values': [1], 'labels': ['Missing'],
                            'marker': dict(colors=['white'], line=dict(color='black', width=1)),
                            'domain': pie_domain, 'hoverinfo': 'skip', 'textinfo': 'none'}
            fig.add_trace(go.Pie(**pie_args))

    # Add axis labels on x-axis (columns);
    fig.add_annotation(x=0.5, y=-0.08, text=column_variable, showarrow=False, xref="paper", yref="paper",
                       font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"), xanchor='center',
                       yanchor='top')
    if additional_column_variable is not None:
        additional_column_variable_map = df.groupby(column_variable, observed=False)[additional_column_variable].agg(
            lambda x: x.unique()[0]).to_dict()
    for col_index, col in enumerate(plate_columns):
        x = (col_index + 0.5) * cell_width
        label_text = f"{col}<br>{additional_column_variable_map.get(col, 'Unknown')}" if additional_row_variable else str(
            col)
        fig.add_annotation(
            x=x, y=-0.01, text=label_text, showarrow=False, xref="paper", yref="paper",
            font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='center', yanchor='top')

    # Add axis labels on y-axis (rows);
    fig.add_annotation(x=-0.08, y=0.5, text=row_variable, showarrow=False, xref="paper", yref="paper",
                       font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"), xanchor='right',
                       yanchor='middle')
    if (additional_row_variable is not None):
        additional_row_variable_map = df.groupby(row_variable, observed=False)[additional_row_variable].agg(
            lambda x: x.unique()[0]).to_dict()
    for row_index, row in enumerate(plate_rows):
        y = 1 - (row_index + 0.5) * cell_height
        label_text = chr(64 + int(row))
        if additional_column_variable:
            label_text += f"<br>{additional_row_variable_map.get(row, 'Unknown')}"
        fig.add_annotation(
            x=-0.01, y=y, text=label_text, showarrow=False, xref="paper", yref="paper",
            font=dict(color="black", family="Times New Roman", size=16, weight="bold"), xanchor='right',
            yanchor='middle')

    # Update layout
    fig.update_layout(
        # title=dict(text=graph_title, y=0.95, yanchor='top', font=dict(size=20)),
        title=dict(text=graph_title, font=dict(color="black", family="Times New Roman", size=26, weight="bold"), x=0,
                   xanchor='left', y=0.97, yanchor='top'),
        # xaxis_title = column_variable,
        # xaxis_title_font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        # yaxis_title = row_variable,
        # yaxis_title_font=dict(color="darkblue", family="Times New Roman", size=20, weight="bold"),
        # xaxis=dict(
        #    title_standoff=60
        # ),
        # yaxis=dict(
        #    title_standoff=60
        # ),
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', font=dict(size=16), itemsizing='constant', tracegroupgap=5,
                    yanchor='top', bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0)'),
        # margin=dict(t=150, b=50, l=50, r=20),
        margin=dict(t=110, b=90, l=130),
        # height = plot_height + 50 + 50,
        # width=plot_width + 70 + 50,
        shapes=[  # add rectangle around pie charts to create plate layout
            dict(type='rect', xref='paper', yref='paper', x0=0, x1=1, y0=0, y1=1, line=dict(color='black', width=2),
                 fillcolor='rgba(0,0,0,0)'),
        ]
    )

    # fig.show(renderer='browser')
    return fig
