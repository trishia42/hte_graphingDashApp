import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, no_update, ctx, dash_table, MATCH
import dash_bootstrap_components as dbc
import io, uuid, base64
from datetime import datetime
import operator as op
import re
import zipfile
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from miscellaneousFunctions import *
from graphingFunctions import *

# Component generation and UI functions
def generate_dropdown_control(dropdown_id, options, label, label_style=None, dropdown_style=None, value=None, multi=False, listbox=False, add_component=None, \
                              disabled=False, clearable=True):
    label_style_ = {'fontSize': '0.8rem', 'marginBottom':'0.1rem'}
    dropdown_style_ = {'fontSize': '0.8rem', 'width': '100%', 'marginBottom': '0.15rem'}
    if label_style:
        label_style_.update(label_style)
    if dropdown_style:
        dropdown_style_.update(dropdown_style)
    row_children = [dbc.Col(dbc.Label(label, html_for=dropdown_id, style=label_style_), width='auto'), dbc.Col()]
    if add_component is not None:
        row_children.append(dbc.Col(add_component))

    dropdown_options = options if (options and isinstance(options[0], dict) and 'label' in options[0] and 'value' in options[0]) else ([{'label': i, 'value': i} for i in options] if options else [])

    return dbc.Form([
        dbc.Row(row_children, align='center'),
        dcc.Dropdown(
            id={'type': 'listbox', 'subtype': dropdown_id} if listbox else dropdown_id,
            #options=[{'label': i, 'value': i} for i in options] if options else [],
            options=dropdown_options,
            value=value,
            multi=multi,
            disabled=disabled,
            clearable=clearable,
            #optionHeight=10,
            maxHeight=300,
            style=dropdown_style_,
        )
    ])

def generate_colorscale_dropdown_options(reverse_option): # Create options with colour previews

    colorscale_options = ['Plotly3', 'Viridis', 'Inferno', 'Magma', 'Plasma', 'Turbo', 'Bluered', 'Electric', 'Hot', 'Jet',
                       'Rainbow', 'Thermal', 'Haline', 'Solar', 'Ice', 'Deep', 'Dense', 'Matter', 'Speed', 'AgSunset',
                       'SunsetDark', 'Aggrnyl', 'Puor', 'RdBu', 'Spectral', 'Balance', 'Delta', 'Curl', 'TealRose', 'Portland',
                       'PRGn', ]

    def colorscale_to_css_gradient(colorscale_name, reverse_option): # to show the colours in the dropdown as well when selecting
        colorscale_name = colorscale_name + '_r' if reverse_option else colorscale_name
        colors = px.colors.get_colorscale(colorscale_name)
        color_stops = ', '.join([f'{c[1]} {int(c[0]*100)}%' for c in colors])
        return f'linear-gradient(to right, {color_stops})'

    colorscale_dropdown_options = [
        {
            'label': html.Div([
                html.Div(cs, style={'width': f'{max(len(cs) for cs in colorscale_options)*0.4}rem', 'marginRight': "0.25rem", 'marginLeft': '0rem', 'whiteSpace': 'nowrap'}),
                html.Div(
                    style={
                        'width': '6.75rem',
                        'height': '1.5rem',
                        'marginLeft': '0.125rem',
                        'marginRight': '0.125rem',
                        'background': colorscale_to_css_gradient(cs, reverse_option),
                        'border': '0.05rem solid #ccc',
                        'borderRadius': '0.05rem',
                    }
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'noWrap', 'maxWidth': 'calc(100% - 1rem)'}),
            'value': cs
        }
        for cs in colorscale_options
    ]

    return colorscale_dropdown_options

category_suffix, plate_variables_columns = '_encoded', ['row', 'column', 'position', 'index', 'well', 'well index', 'id']
bootstrap_cols = 12
sidebar_width = (1/6)*bootstrap_cols # in bootstrap definition
graph_options = {'Parallel Coordinates': 0, 'Scatter': 1, 'Heatmap': 2, 'Pie Charts': 3, 'Bar Chart': 4, 'Dumbbell Treillis': 5}
graph_type_label_style = {'fontSize': '0.92rem', 'fontWeight': 'bold', 'marginTop':'0.75rem', 'marginBottom':'0.25rem'}
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
colorscale_dropdown_options = generate_colorscale_dropdown_options(False) # for initialization

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, assets_folder='assets')
server=app.server # for Render deploy - this is for Render hosting
app.title = 'Data Visualization Dashboard'

# Actual app layout - I hate html and css so sorry
app.layout = dbc.Container([
    dcc.Store(id='dataframe-store'),
    dcc.Store(id='df-original-columns-store'),
    dcc.Store(id='df-row-variable-store'),
    dcc.Store(id='df-column-variable-store'),
    dcc.Store(id='df-numeric-columns-store'),
    dcc.Store(id='df-categorical-columns-store'),
    dcc.Store(id='datatable-modified-store'),
    dcc.Store(id='raw-figures-store'),
    dbc.Row([
        dbc.Col([
            generate_dropdown_control('graph-selection', list(graph_options.keys()), 'Type of Graph', label_style={'fontSize': '0.92rem', 'fontWeight': 'bold', 'marginTop': '0.375rem', 'marginBottom':'0.35rem'}, \
                                      dropdown_style={'fontSize': '0.875rem'}, value='Parallel Coordinates', clearable=False),
            html.Div(id='parallel-options', style={'display': 'block'}, children=[
                html.H6('Parallel Coordinates Options', style=graph_type_label_style),

                html.Div(id='parallel-variables-wrapper', children=[
                    generate_dropdown_control('parallel-variables', None, 'Variables', multi=True, listbox=True)
                ]),
                html.Div(id='parallel-color-variable-wrapper', children=[
                    generate_dropdown_control('parallel-color-variable', None, 'Color Variable', value=None)
                ], style={'marginBottom': '0.5rem'}),
            ]),
            html.Div(id='scatter-options', style={'display': 'none'}, children=[
                html.H6('Scatter Plot Options', style=graph_type_label_style),
                html.Div(id='scatter-x-variable-wrapper', children=[
                    generate_dropdown_control('scatter-x-variable', None, 'X Variable'),
                ]),
                html.Div(id='scatter-y-variable-wrapper', children=[
                    generate_dropdown_control('scatter-y-variable', None, 'Y Variable'),
                ]),
                html.Div(id='scatter-x-subplots-variables-wrapper', children=[
                    generate_dropdown_control('scatter-x-subplots-variables', None, 'X Variables (subplots)', multi=True, listbox=True),
                ]),
                html.Div(id='scatter-z-variable-wrapper', children=[
                    generate_dropdown_control('scatter-z-variable', None, 'Z Variable (for 3D)', add_component=dbc.Checkbox(id='scatter-surface', label='Surface', style={'fontSize':'0.8rem', 'fontStyle': 'italic', 'marginRight': '0.25rem', 'marginTop':'0.35rem', 'marginBottom':'0rem'})),
                ]),
                html.Div(id='scatter-size-variable-wrapper', children=[
                    generate_dropdown_control('scatter-size-variable', None, 'Size Variable'),
                ]),
                html.Div(id='scatter-symbol-variable-wrapper', children=[
                    generate_dropdown_control('scatter-symbol-variable', None, 'Symbol Variable'),
                ]),
                html.Div(id='scatter-color-variable-wrapper', children=[
                    generate_dropdown_control('scatter-color-variable', None, 'Color Variable'),
                ], style={'marginBottom': '0.5rem'}),
            ]),
            html.Div(id='heatmap-options', style={'display': 'none'}, children=[
                html.H6('Heatmap Options [plate]', style=graph_type_label_style),
                html.Div(id='heatmap-color-variable-wrapper', children=[
                    generate_dropdown_control('heatmap-color-variable', None, 'Color Variable'),
                ]),
                html.Div(id='heatmap-add-row-variable-wrapper', children=[
                    generate_dropdown_control('heatmap-add-row-variable', None, 'Additional Row Variable', multi=True, listbox=True),
                ]),
                html.Div(id='heatmap-add-column-variable-wrapper', children=[
                    generate_dropdown_control('heatmap-add-column-variable', None, 'Additional Column Variable', multi=True, listbox=True),
                ]),
                html.Div(
                    generate_dropdown_control('heatmap-smooth-option', ['False', 'best', 'fast'], 'Smooth', value='False', ),
                    style={'marginBottom': '0.5rem'}
                ),
            ]),
            html.Div(id='piechart-options', style={'display': 'none'}, children=[
                html.H6('Pie Chart Options [plate]', style=graph_type_label_style),
                html.Div(id='piechart-variables-wrapper', children=[
                    generate_dropdown_control('piechart-variables', None, 'Variables', multi=True, listbox=True),
                ]),
                html.Div(id='piechart-add-row-variable-wrapper', children=[
                    generate_dropdown_control('piechart-add-row-variable', None, 'Additional Row Variable', multi=True, listbox=True)
                ]),
                html.Div(id='piechart-add-column-variable-wrapper', children=[
                    generate_dropdown_control('piechart-add-column-variable', None, 'Additional Column Variable', multi=True, listbox=True)
                ]),
                generate_dropdown_control('piechart-normalization-type', ['Normalize to 100%', 'Values are %', 'Normalize to value'], 'Normalization', value='Normalize to 100%', clearable=False),
                dbc.Label('Value (if applicable):', style={'fontSize': '0.8rem', 'marginBottom':'0.2rem'}),
                dbc.Input(id='piechart-normalization-value', type='number', placeholder='Enter value', style={'fontSize':'0.8rem'}),
                html.Div(id='piechart-add-options-wrapper', children=[
                    dbc.Row([
                        dbc.Col(dbc.Checkbox(id='piechart-donut', label='Donut', value=False, style={'fontSize': '0.8rem', 'marginTop': '0.25rem', 'marginRight': '0.25rem'}), width='auto', align='left'),
                        dbc.Col(dbc.Checkbox(id='piechart-cakeplots', label='Cake plots', value=True, style={'fontSize': '0.8rem', 'marginTop': '0.25rem', 'marginRight': '0.25rem'}), width='auto', align='left')
                    ]),
                ])
            ]),
            html.Div(id='barchart-options', style={'display': 'none'}, children=[
                html.H6('Bar Chart Options', style=graph_type_label_style),
                html.Div(id='barchart-x-variable-wrapper', children=[
                    generate_dropdown_control('barchart-x-variable', None, 'X Variable', value=None),
                ]),
                html.Div(id='barchart-variables-wrapper', children=[
                    generate_dropdown_control('barchart-variables', None, 'Y Variables (same units)', multi=True, listbox=True),
                ]),
                html.Div(id='barchart-pattern-variable-wrapper', children=[
                    generate_dropdown_control('barchart-pattern-variable', None, 'Bar Pattern Variable', value=None),
                ]),
                html.Div(id='barchart-group-variables-by-wrapper', children=[
                    generate_dropdown_control('barchart-group-variables-by', None, 'Group by')
                ]),
                html.Div(id='barchart-barmode-option-wrapper', children=[
                    generate_dropdown_control('barchart-barmode-option', ['Group', 'Stack', 'Overlay'], 'Barmode', value='Group', clearable=False),
                ], style={'marginBottom': '0.5rem'}),
            ]),
            html.Div(id='dumbbell-options', style={'display': 'none'}, children=[
                html.H6('Dumbbell Treillis Options', style=graph_type_label_style),
                html.Div(id='dumbbell-x-variable-wrapper', children=[
                    generate_dropdown_control('dumbbell-x-variable', None, 'X Variable', value=None),
                ]),
                html.Div(id='dumbbell-y-variable-wrapper', children=[
                    generate_dropdown_control('dumbbell-y-variable', None, 'Y Variable', value=None),
                ]),
                html.Div(id='dumbbell-color-variable-wrapper', children=[
                    generate_dropdown_control('dumbbell-color-variable', None, 'Color Variable', value=None),
                ]),
                html.Div(id='dumbbell-symbol-variable-wrapper', children=[
                    generate_dropdown_control('dumbbell-symbol-variable', None, 'Symbol Variable', value=None),
                ]),
                html.Div(id='dumbbell-grouped-variables-wrapper', children=[
                    generate_dropdown_control('dumbbell-grouped-variables', None, 'Additional X Grouped Over', multi=True, listbox=True)
                ], style={'marginBottom': '0.5rem'}),
            ]),
            html.Hr(style={'marginBottom': '0rem', 'marginTop':'0rem'}),
            html.Div([
                dbc.Row([
                    dbc.Col(
                        generate_dropdown_control('colorscale', colorscale_dropdown_options, 'Color Scale', value=colorscale_dropdown_options[0]['value'], clearable=False), width=10, \
                        style={'paddingRight': '0.3rem', 'marginRight': '0rem'}
                    ),
                    dbc.Col(
                        dbc.Checkbox(id='colorscale-reverse', label='_r', value=False, label_style={'verticalAlign':'bottom', 'paddingBottom':'0rem', 'marginBottom':'0rem'}), width=2, \
                        style={'fontSize': '1.0rem', 'display':'flex', 'alignItems':'flex-end', 'justifyContent': 'flex-start', 'paddingRight': '0', 'paddingLeft': '0'}
                    ),
                ], style={'marginBottom':'0.2rem'}), # added
            ]),
            html.Div(id='split-by-variable-wrapper', children=[
                generate_dropdown_control('split-by-variable', None, 'Split by', value=None)
            ]),
            html.Div(id='plate-rows-as-alpha-wrapper', children=[
                dbc.Checkbox(id='plate-rows-as-alpha', label='Rows as alpha', value=True,  style={'fontSize':'0.8rem', 'marginTop': '0.15rem', 'marginBottom': '0rem', 'marginLeft': '0.25rem'}),
            ]),
            dbc.Label('Multiple dataframes handling', style={'fontSize': '0.875rem', 'marginTop':'0rem', 'marginBottom':'0rem'}),

            html.Div(id='multiple-dataframes-handling-wrapper', children=[
                dbc.Row([
                    dbc.Col(
                        generate_dropdown_control('multiple-dataframes-id-column', None, 'ID-Column:', disabled=False), width=9, style={'paddingLeft': '0.85rem', 'paddingRight':'0.15rem'}
                    ),
                    dbc.Col(
                        html.Div(
                            dcc.RadioItems(id='multiple-dataframes-handling', options=[{'label': 'First', 'value': 'First'}, {'label': 'All', 'value': 'All'}], value='First', inline=False, inputStyle={'margin-right': '0.15rem'}, labelStyle={'marginBottom': '0rem', 'fontSize': '0.8rem'}
                        ), style={'paddingTop': '1.5rem', 'display': 'flex', 'alignItems': 'center'})# or 'flex-end'
                    ),
                ], align = 'center', style={'marginBottom':'0rem', 'marginTop':'0rem'}), # added ,
                dbc.Row([
                    dbc.Col(
                        dbc.Checkbox(id='multiple-dataframes-reverse', label='Reverse', value=False), width='auto',
                    )
                ], style={'marginRight': '0rem', 'paddingLeft': '0.25rem', 'fontSize': '0.8rem', 'marginBottom': '0rem'}),
            ], style={'marginBottom': '0rem'}),
            html.Div(id='multiple-series-names-wrapper', children=[
                dbc.Label('Multi series names:', style={'marginTop': '0rem', 'marginBottom': '0.1rem', 'fontSize':'0.8rem'}),
                dbc.Input(id='multiple-series-names', type='text', placeholder='Series 1|Series 2', disabled=True, style={'marginBottom': '0.15rem', 'fontSize':'0.8rem'}),
            ], style={'marginTop': '-0.5rem', 'marginBottom': '0rem'}),
            dbc.Label('Graph Title(s)', style={'marginTop':'0rem', 'marginBottom':'0.1rem', 'fontSize':'0.8rem'}),
            dbc.Input(id='graph-title', type='text', placeholder='Title1|Title2', style={'marginBottom': '1rem', 'fontSize': '0.8rem'}),
            dbc.Row([
                dbc.Col(dbc.Button('Test Data', id='test-data', color='info', className='btn-custom-press', size='sm', style={'display':'flex', 'width': '6.5rem', 'height':'2rem', 'justifyContent':'center', 'alignItems':'center', 'boxShadow': '2px 2px 4px 2px rgba(0, 0, 0, 0.6)'})),
                dbc.Col(dbc.Button('Generate', id='generate-button', color='success', className='btn-custom-press', size='sm', style={'display':'flex', 'width': '6.5rem', 'height':'2rem', 'justifyContent':'center', 'alignItems':'center', 'boxShadow': '2px 2px 4px 2px rgba(0, 0, 0, 0.6)'})), # changed button config)
            ], className='p-0 g-3', style={'marginLeft':'0.25rem', 'marginBottom':'0.65rem'}),
            dbc.Row([
                dbc.Col(dcc.Upload(id='upload-data', children=dbc.Button('Upload File', color='primary', size='sm', className='btn-custom-press', style={'display':'flex', 'width': '6.5rem', 'height':'2rem', 'justifyContent':'center', 'alignItems':'center', 'boxShadow': '2px 2px 4px 2px rgba(0, 0, 0, 0.6)'}), multiple=False)),
                dbc.Col(dbc.Button('Download', id='download-button', color='secondary', className='btn-custom-press', size='sm', style={'display':'flex', 'width': '6.5rem', 'height':'2rem', 'justifyContent':'center', 'alignItems':'center', 'boxShadow': '2px 2px 4px 2px rgba(0, 0, 0, 0.6)'})),
                dcc.Download(id='download-zip'),
            ], className='p-0 g-3', style={'marginLeft':'0.25rem', 'marginTop':'0ren'}),
            html.Div(id='upload-output', style={'padding': '0.625rem', 'marginLeft':'0.5rem', 'fontSize':'0.8rem'}),
        ], width=sidebar_width, style={'borderRight': '0.075rem solid #ddd', 'overflowY': 'scroll', 'height':'100vh', 'scrollbarwidth':'none', 'msOverflowStyle':'none'}),
        dbc.Col([
            html.Div(id='app-note', children=[
                'App will spin down after 15 minutes of inactivity, requiring another 50-60 seconds to reactivate - best viewed on standard screens.  This is a work in progress; please report any issues/bugs (and I expect a lot of them) at the ',
                html.A('GitHub Repository', href='https://github.com/trishia42/hte_graphingDashApp', target='_blank', style={'color': '#1f77b4', 'textDecoration': 'underline'}), '.'
                ],
                style={'backgroundColor': '#f8f9fa','padding': '0.125rem 0.125rem', 'marginBottom': '0.3rem', 'marginTop': '0.3rem', 'fontSize': '0.75rem', 'width': '100%','fontStyle':'italic'}),

            html.Div(id='status-bar', children='Status', style={'backgroundColor': '#f8f9fa', 'border': '0.075rem solid #ccc','padding': '0.5rem 0.75rem', 'marginBottom': '0.5rem', 'marginTop': '0.3rem', \
                    'fontWeight': 'bold', 'fontSize': '0.875rem', 'width': '100%',}),

            dbc.Row([
                html.Div(dcc.Loading(html.Div(id='graph-container', style={'display':'none'}))),
            ]),
            dbc.Row([
                html.Hr(),
                html.Div(id='datatable-container', style={'display': 'none', 'marginTop':'2rem'}, children=[
                    html.H5('Current Data Table', style={'fontSize':'0.92rem', 'fontWeight':'bold'}),
                    dash_table.DataTable(id='datatable', data=[], columns=[], page_size=10,style_table={'overflowX': 'auto', 'border': '0.05rem solid lightgrey', 'fontsize':'0.8rem'},
                                        style_cell={'textAlign': 'left', 'padding': '0.3rem', 'whiteSpace': 'normal', 'height': 'auto', 'border': '0.05rem solid lightgrey'},
                                        style_header={'fontWeight': 'bold', 'border': '0.05rem solid lightgrey', 'fontSize':'0.85rem'}, style_cell_conditional=[], editable=True, sort_action='native',
                                        filter_action='native', row_deletable=True),
                ]),
            ]),
            html.Div(
                id='conversion-container', children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label('Column ID', html_for='add-row-column-col-id', style={'fontSize': '0.85rem', 'fontFamily': 'Arial, sans-serif'}),
                            dbc.Input(id='add-row-column-col-id', type='text', placeholder='', style={'marginBottom': '0.5rem'}),
                        ], width='1'),
                        dbc.Col([
                            dbc.Label('Plate Rows', html_for='add-row-column-plate-rows', style={'fontSize': '0.85rem', 'fontFamily': 'Arial, sans-serif'}),
                            dbc.Input(id='add-row-column-plate-rows', type='number', step=1, placeholder='', style={'marginBottom': '0.5rem'}),
                        ], width='1'),
                        dbc.Col([
                            dbc.Label('Plate Columns', html_for='add-row-column-plate-columns', style={'fontSize': '0.85rem', 'fontFamily': 'Arial, sans-serif'}),
                            dbc.Input(id='add-row-column-plate-columns', type='number', step=1, placeholder='', style={'marginBottom': '0.5rem'}),
                        ], width='1'),
                        dbc.Col([
                            dbc.Button('Row/Column', id='add-row-column-button', color='primary', size='sm', className='btn-custom-press', outline=True, style={'display':'flex', 'width': '6.5rem', 'height':'2rem', 'justifyContent':'center', \
                                'alignItems':'center', 'marginTop':'2rem', 'marginLeft':'0.5rem', 'boxShadow': '2px 2px 2px 2px rgba(0, 0, 0, 0.2)'}),
                        ], width='auto')
                    ])
            ]),
        ], width=(bootstrap_cols - sidebar_width),)
    ])
], fluid=True)

number_of_UI_interactivity_outputs = 28
#<editor-fold desc="**app.callback => UI Interactivity">
@app.callback(
    Output('parallel-options', 'style'),
    Output('scatter-options', 'style'),
    Output('heatmap-options', 'style'),
    Output('piechart-options', 'style'),
    Output('barchart-options', 'style'),
    Output('dumbbell-options', 'style'),
    Output('colorscale', 'options', allow_duplicate=True),
    Output('colorscale', 'value', allow_duplicate=True),

    Output('multiple-dataframes-id-column', 'disabled', allow_duplicate=True),
    Output('multiple-dataframes-id-column', 'value', allow_duplicate=True),
    Output('scatter-x-variable', 'value', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'scatter-x-subplots-variables'}, 'value', allow_duplicate=True),
    Output('scatter-z-variable', 'value', allow_duplicate=True),
    Output('multiple-series-names', 'value', allow_duplicate=True),
    Output('multiple-series-names', 'disabled', allow_duplicate=True),

    Output('heatmap-add-row-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-row-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-row-variable'}, 'value', allow_duplicate=True),
    Output('heatmap-add-column-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-column-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-column-variable'}, 'value', allow_duplicate=True),

    Output('piechart-add-row-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-row-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-row-variable'}, 'value', allow_duplicate=True),
    Output('piechart-add-column-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-column-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-column-variable'}, 'value', allow_duplicate=True),

    Output('status-bar', 'children', allow_duplicate=True),

    Input('graph-selection', 'value'),
    State('colorscale', 'value'),
    Input('colorscale-reverse', 'value'),
    State('dataframe-store', 'data'),
    State('multiple-dataframes-id-column', 'options'),
    Input('scatter-x-variable', 'value'),
    Input({'type': 'listbox', 'subtype': 'scatter-x-subplots-variables'}, 'value'),
    Input('scatter-z-variable', 'value'),
    Input('multiple-dataframes-id-column', 'value'),
    Input('multiple-dataframes-handling', 'value'),
    State('df-row-variable-store', 'data'),
    State('df-column-variable-store', 'data'),
    State('df-categorical-columns-store', 'data'),
    prevent_initial_call = True
)
#</editor-fold>

def UIInteractivity(graph_selection, colorscale_name, colorscale_reverse, stored_dataframe, multiple_dataframes_id_column_options, scatter_x_variable, scatter_x_subplots_variables, \
                    scatter_z_variable, multiple_dataframes_id_column_value, multiple_dataframes_handling, df_row_variable, df_column_variable, df_categorical_columns):

    #parallel_output, scatter_output, heatmap_output, piechart_output, barchart_output, dumbbell_output, colorscale_dropdown_options, colorscale_dropdown_value, multiple_dataframes_id_column_state, multiple_dataframes_id_column_value, \
    #    scatter_x_variable_output, scatter_x_subplots_variables_output, scatter_z_variable_output, multi_series_names_value_output, multi_series_names_disabled_output, additional_row_columns_list, \
    #    additional_column_columns_list = [dash.no_update]*number_of_UI_interactivity_outputs

    parallel_output, scatter_output, heatmap_output, piechart_output, barchart_output, dumbbell_output, colorscale_dropdown_options, colorscale_dropdown_value, multiple_dataframes_id_column_state, \
        multiple_dataframes_id_column_value,scatter_x_variable_output, scatter_x_subplots_variables_output, scatter_z_variable_output, multi_series_names_value_output, multi_series_names_disabled_output, \
        additional_row_columns_list, additional_column_columns_list = [dash.no_update]*(number_of_UI_interactivity_outputs - 11)

    additional_row_column_changed, status_output = False, dash.no_update

    try:
        trigger = ctx.triggered_id
        if trigger == 'colorscale-reverse':
            colorscale_dropdown_value = colorscale_name
            colorscale_dropdown_options = generate_colorscale_dropdown_options(colorscale_reverse)
        elif trigger == 'graph-selection':
            def show(name):
                return {'display': 'block'} if graph_selection == name else {'display': 'none'}
            parallel_output = show('Parallel Coordinates')
            scatter_output = show('Scatter')
            heatmap_output = show('Heatmap')
            piechart_output = show('Pie Charts')
            barchart_output = show('Bar Chart')
            dumbbell_output = show('Dumbbell Treillis')

            if graph_selection in ['Heatmap', 'Pie Charts']:
                multiple_dataframes_id_column_state = True
                if multiple_dataframes_id_column_options and stored_dataframe:
                    multiple_dataframes_id_column_value = 'Row/Column' if any(opt['value'] == 'Row/Column' for opt in multiple_dataframes_id_column_options) else None
                    row_consistent_columns, column_consistent_columns = get_consistent_columns(next(iter((split_dataframe(pd.read_json(io.StringIO(stored_dataframe['data']), orient='split'), multiple_dataframes_id_column_value)).values())) if multiple_dataframes_handling == 'First' else pd.read_json(io.StringIO(stored_dataframe['data']), orient='split').copy(), df_categorical_columns, df_row_variable, df_column_variable)
                    additional_row_columns_list, additional_column_columns_list = [{'label': option, 'value': option} for option in row_consistent_columns], [{'label': option, 'value': option} for option in column_consistent_columns]
                    additional_row_column_changed = True
            else:
                multiple_dataframes_id_column_state = False
                multiple_dataframes_id_column_value = dash.no_update
        elif trigger == 'scatter-x-variable' or trigger == 'scatter-z-variable':
            if scatter_x_variable is not None or scatter_z_variable is not None:
                scatter_x_subplots_variables_output = None
        elif isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get('subtype') == 'scatter-x-subplots-variables':
            if scatter_x_subplots_variables is not None:
                scatter_x_variable_output, scatter_z_variable_output = None, None
        elif trigger == 'multiple-dataframes-id-column' or trigger == 'multiple-dataframes-handling':
            if multiple_dataframes_id_column_value is not None and multiple_dataframes_handling == 'All':
                multi_series_names_value_output, multi_series_names_disabled_output = dash.no_update, False
            else:
                multi_series_names_value_output, multi_series_names_disabled_output = None, True
            if multiple_dataframes_id_column_value is not None and graph_selection in ['Heatmap', 'Pie Charts'] and stored_dataframe:
                row_consistent_columns, column_consistent_columns = get_consistent_columns(next(iter((split_dataframe(pd.read_json(io.StringIO(stored_dataframe['data']), orient='split'), multiple_dataframes_id_column_value)).values())) if multiple_dataframes_handling == 'First' else pd.read_json(io.StringIO(stored_dataframe['data']), orient='split').copy(), df_categorical_columns, df_row_variable, df_column_variable)
                additional_row_columns_list, additional_column_columns_list = [{'label': option, 'value': option} for option in row_consistent_columns], [{'label': option, 'value': option} for option in column_consistent_columns]
                additional_row_column_changed = True

        return parallel_output, scatter_output, heatmap_output, piechart_output, barchart_output, dumbbell_output, colorscale_dropdown_options, colorscale_dropdown_value, multiple_dataframes_id_column_state, \
            multiple_dataframes_id_column_value, scatter_x_variable_output, scatter_x_subplots_variables_output, scatter_z_variable_output, multi_series_names_value_output, multi_series_names_disabled_output, \
            str(uuid.uuid4()) if additional_row_column_changed else dash.no_update, additional_row_columns_list, [] if additional_row_column_changed else dash.no_update, \
            str(uuid.uuid4()) if additional_row_column_changed else dash.no_update, additional_column_columns_list, [] if additional_row_column_changed else dash.no_update, \
            str(uuid.uuid4()) if additional_row_column_changed else dash.no_update, additional_row_columns_list, [] if additional_row_column_changed else dash.no_update, \
            str(uuid.uuid4()) if additional_row_column_changed else dash.no_update, additional_column_columns_list, [] if additional_row_column_changed else dash.no_update, status_output

    except Exception as e:
        exception_message = generate_exception_message(e)
        return [dash.no_update] * (number_of_UI_interactivity_outputs  - 1) + [exception_message]

number_of_load_data_outputs = 66 + 42
#<editor-fold desc="**app.callback => Handle uploads/test data and programmatic changes to data table">
@app.callback(
    Output('dataframe-store', 'data', allow_duplicate=True),
    Output('upload-output', 'children', allow_duplicate=True),
    Output('df-original-columns-store', 'data', allow_duplicate=True),
    Output('df-row-variable-store', 'data', allow_duplicate=True),
    Output('df-column-variable-store', 'data', allow_duplicate=True),
    Output('df-numeric-columns-store', 'data', allow_duplicate=True),
    Output('df-categorical-columns-store', 'data', allow_duplicate=True),

    Output('parallel-variables-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'parallel-variables'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'parallel-variables'}, 'value', allow_duplicate=True),
    Output('parallel-color-variable-wrapper', 'key', allow_duplicate=True),
    Output('parallel-color-variable', 'options', allow_duplicate=True),
    Output('parallel-color-variable', 'value', allow_duplicate=True),

    Output('scatter-x-variable-wrapper', 'key', allow_duplicate=True),
    Output('scatter-x-variable', 'options', allow_duplicate=True),
    Output('scatter-x-variable', 'value', allow_duplicate=True),
    Output('scatter-x-subplots-variables-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'scatter-x-subplots-variables'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'scatter-x-subplots-variables'}, 'value', allow_duplicate=True),
    Output('scatter-y-variable-wrapper', 'key', allow_duplicate=True),
    Output('scatter-y-variable', 'options', allow_duplicate=True),
    Output('scatter-y-variable', 'value', allow_duplicate=True),
    Output('scatter-z-variable-wrapper', 'key', allow_duplicate=True),
    Output('scatter-z-variable', 'options', allow_duplicate=True),
    Output('scatter-z-variable', 'value', allow_duplicate=True),
    Output('scatter-surface', 'value', allow_duplicate=True),
    Output('scatter-size-variable-wrapper', 'key', allow_duplicate=True),
    Output('scatter-size-variable', 'options', allow_duplicate=True),
    Output('scatter-size-variable', 'value', allow_duplicate=True),
    Output('scatter-symbol-variable-wrapper', 'key', allow_duplicate=True),
    Output('scatter-symbol-variable', 'options', allow_duplicate=True),
    Output('scatter-symbol-variable', 'value', allow_duplicate=True),
    Output('scatter-color-variable-wrapper', 'key', allow_duplicate=True),
    Output('scatter-color-variable', 'options', allow_duplicate=True),
    Output('scatter-color-variable', 'value', allow_duplicate=True),

    Output('heatmap-color-variable-wrapper', 'key', allow_duplicate=True),
    Output('heatmap-color-variable', 'options', allow_duplicate=True),
    Output('heatmap-color-variable', 'value', allow_duplicate=True),
    Output('heatmap-add-row-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-row-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-row-variable'}, 'value', allow_duplicate=True),
    Output('heatmap-add-column-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-column-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-column-variable'}, 'value', allow_duplicate=True),

    Output('piechart-donut', 'value', allow_duplicate=True),
    Output('piechart-variables-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-variables'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-variables'}, 'value', allow_duplicate=True),
    Output('piechart-add-row-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-row-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-row-variable'}, 'value', allow_duplicate=True),
    Output('piechart-add-column-variable-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-column-variable'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-column-variable'}, 'value', allow_duplicate=True),

    Output('barchart-x-variable-wrapper', 'key', allow_duplicate=True),
    Output('barchart-x-variable', 'options', allow_duplicate=True),
    Output('barchart-x-variable', 'value', allow_duplicate=True),
    Output('barchart-variables-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'barchart-variables'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'barchart-variables'}, 'value', allow_duplicate=True),

    Output('barchart-pattern-variable-wrapper', 'key', allow_duplicate=True),
    Output('barchart-pattern-variable', 'options', allow_duplicate=True),
    Output('barchart-pattern-variable', 'value', allow_duplicate=True),

    Output('barchart-group-variables-by-wrapper', 'key', allow_duplicate=True),
    Output('barchart-group-variables-by', 'options', allow_duplicate=True),
    Output('barchart-group-variables-by', 'value', allow_duplicate=True),
    Output('barchart-barmode-option-wrapper', 'key', allow_duplicate=True),
    Output('barchart-barmode-option', 'options', allow_duplicate=True),
    Output('barchart-barmode-option', 'value', allow_duplicate=True),

    Output('dumbbell-x-variable-wrapper', 'key', allow_duplicate=True),
    Output('dumbbell-x-variable', 'options', allow_duplicate=True),
    Output('dumbbell-x-variable', 'value', allow_duplicate=True),
    Output('dumbbell-y-variable-wrapper', 'key', allow_duplicate=True),
    Output('dumbbell-y-variable', 'options', allow_duplicate=True),
    Output('dumbbell-y-variable', 'value', allow_duplicate=True),
    Output('dumbbell-grouped-variables-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'dumbbell-grouped-variables'}, 'options', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'dumbbell-grouped-variables'}, 'value', allow_duplicate=True),
    Output('dumbbell-color-variable-wrapper', 'key', allow_duplicate=True),
    Output('dumbbell-color-variable', 'options', allow_duplicate=True),
    Output('dumbbell-color-variable', 'value', allow_duplicate=True),
    Output('dumbbell-symbol-variable-wrapper', 'key', allow_duplicate=True),
    Output('dumbbell-symbol-variable', 'options', allow_duplicate=True),
    Output('dumbbell-symbol-variable', 'value', allow_duplicate=True),

    Output('split-by-variable-wrapper', 'key', allow_duplicate=True),
    Output('split-by-variable', 'options', allow_duplicate=True),
    Output('split-by-variable', 'value', allow_duplicate=True),
    Output('multiple-dataframes-handling-wrapper', 'key', allow_duplicate=True),
    Output('multiple-dataframes-handling', 'options', allow_duplicate=True),
    Output('multiple-dataframes-handling', 'value', allow_duplicate=True),
    Output('multiple-dataframes-id-column', 'options', allow_duplicate=True),
    Output('multiple-dataframes-id-column', 'value', allow_duplicate=True),
    Output('multiple-dataframes-id-column', 'disabled', allow_duplicate=True),
    Output('multiple-series-names-wrapper', 'key', allow_duplicate=True),
    Output('multiple-series-names', 'value', allow_duplicate=True),

    Output('datatable', 'data', allow_duplicate=True),
    Output('datatable', 'columns', allow_duplicate=True),
    Output('datatable-container', 'style', allow_duplicate=True),
    Output('datatable', 'style_cell_conditional', allow_duplicate=True),
    Output('datatable', 'sort_by', allow_duplicate=True),
    Output('datatable', 'filter_query', allow_duplicate=True),
    Output('graph-container', 'children', allow_duplicate=True),
    Output('graph-container', 'style', allow_duplicate=True),
    Output('raw-figures-store', 'data', allow_duplicate=True),
    Output('datatable-modified-store', 'data', allow_duplicate=True),
    Output('datatable', 'data_timestamp', allow_duplicate=True),
    Output('upload-data', 'contents', allow_duplicate=True),
    Output('status-bar', 'children', allow_duplicate=True),

    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('test-data', 'n_clicks'),
    Input('add-row-column-button', 'n_clicks'),
    Input('datatable', 'data_timestamp'),
    Input('datatable', 'sort_by'),
    Input('datatable', 'filter_query'),
    State('datatable', 'columns'),
    State('datatable', 'data'),
    State('add-row-column-col-id', 'value'),
    State('add-row-column-plate-rows', 'value'),
    State('add-row-column-plate-columns', 'value'),
    State('graph-selection', 'value'),
    prevent_initial_call=True
)
#</editor-fold>

def load_data(upload_contents, load_filename, n_clicks_test_data, n_clicks_add_row_column, datatable_timestamp, datatable_sort_by, datatable_filter_query, datatable_columns, datatable_data, \
              add_row_column_col_id, add_row_column_plate_rows, add_row_column_plate_columns, graph_selection):

    # Note; we are updating datatable_timestamp with itself since otherwise it would not trigger graph callback on changes
    trigger, datatable_modified, sort_by_output, filter_query_output, new_df, status = None, False, dash.no_update, dash.no_update, False, '' # on a new load
    trigger = ctx.triggered_id
    try:
        if trigger == 'upload-data' or trigger == 'test-data':
            if trigger == 'upload-data':
                if upload_contents is None:
                    return [dash.no_update] * (number_of_load_data_outputs - 2) + ['No upload contents.']
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                data_to_load = io.BytesIO(decoded)
            elif trigger == 'test-data':
                if n_clicks_test_data is None or n_clicks_test_data == 0:
                    return [dash.no_update] * (number_of_load_data_outputs)
                data_to_load, load_filename = 'testData.xlsx', 'testData.xlsx'
            try:
                if load_filename.endswith('.xlsx') or load_filename.endswith('.xls'):
                    df = pd.read_excel(data_to_load, na_values=[''], keep_default_na=False)
                elif load_filename.endswith('.csv'):
                    df = pd.read_csv(data_to_load, na_values=[''], keep_default_na=False)
                else:
                    return [dash.no_update] * (number_of_load_data_outputs - 1) + ['Only .xlsx/xls and .csv files are currently supported.']

            except Exception as e:
                return [dash.no_update] * (number_of_load_data_outputs - 1) + [f'Error processing file: {e}']

            new_df, status = True, 'File successfully loaded - '
            sort_by_output, filter_query_output = [], '' # resetting on new load
        elif trigger == 'add-row-column-button':
            if n_clicks_add_row_column is None or n_clicks_add_row_column == 0 or not datatable_data:
                return [dash.no_update] * number_of_load_data_outputs
            df = pd.DataFrame(datatable_data)
            if 'Row' in df.columns or 'Column' in df.columns:
                return [dash.no_update] * (number_of_load_data_outputs - 1) + ['Row and/or Column already exist.']

            if add_row_column_col_id is not None and add_row_column_plate_rows is not None and add_row_column_plate_columns is not None:
                # note we can refer to a column by index with df.iloc[:, 1] (this would be the second column, 0-based); the index of a column with its name by using df.columns.get_loc(col_choice)
                # and we can get the name of a column with its index with df.columns[index]
                try:
                    conversion_column_id_int = int(add_row_column_col_id)
                    add_row_column_col_id = df.columns[conversion_column_id_int]
                except ValueError: # not an int, treat as text
                    if add_row_column_col_id not in df.columns:
                        return [dash.no_update] * (number_of_load_data_outputs - 1) + ['Specified conversion column not found in dataframe.']
                if not is_numeric_dtype(df[add_row_column_col_id]):
                    return [dash.no_update] * (number_of_load_data_outputs - 1) + ['Specified conversion column is not numeric.']

                rows, cols = [], []
                for idx, val in enumerate(df[add_row_column_col_id]):
                    if int(val) % int(add_row_column_plate_columns) > 0:
                        row = int(val) // int(add_row_column_plate_columns) + 1
                        col = int(val) % int(add_row_column_plate_columns)
                    else:
                        row = int(val) // int(add_row_column_plate_columns)
                        col = int(add_row_column_plate_columns)
                    rows.append(row)
                    cols.append(col)
                column_index = df.columns.get_loc(add_row_column_col_id)
                df.insert(column_index + 1, 'Row', rows)
                df.insert(column_index + 2, 'Column', cols)
                datatable_modified = True
            else:
                return [dash.no_update] * (number_of_load_data_outputs - 1) + ['Missing information for conversion.']

        elif trigger == 'datatable':
            if not datatable_data:
                return [dash.no_update] * number_of_load_data_outputs
            df = pd.DataFrame(datatable_data)[[col['id'] for col in datatable_columns]]
            if datatable_sort_by:
                for s in reversed(datatable_sort_by):
                    df = df.sort_values(s['column_id'], ascending=s['direction'] == 'asc', kind='stable')
            if datatable_filter_query:
                filtering_expressions = datatable_filter_query.split(' && ')
                for expression in filtering_expressions:
                    if not expression:
                        continue
                    match = re.match(r'\{([^}]+)\}\s*(s?|i?)(>=|<=|!=|=|>|<|contains)\s*(?:\'|")?(.*?)(?:\'|")?$', expression.strip())
                    if match:
                        column, operator_prefix, operator, value = match.group(1).strip(), match.group(2), match.group(3).strip(), match.group(4).strip('\'" ')
                        if pd.api.types.is_numeric_dtype(df[column]):
                            try:
                                value = float(value)
                            except ValueError:
                                continue
                        ops = {'=': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}
                        if operator in ops:
                            df = df[ops[operator](df[column], value)]
                        elif (operator == 'contains'):
                            if operator_prefix == 's':  # case-sensitive
                                df = df[df[column].astype(str).str.contains(value, case=True)]
                            else:  # case-insensitive (operator_prefix = i or no operator_prefix)
                                df = df[df[column].astype(str).str.lower().str.contains(value.lower(), na=False)]
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass  # leave as is
            datatable_modified = True  # something changed
        else:
            return [dash.no_update] * number_of_load_data_outputs

        # Drop empty lines (removes blank rows) and columns without headers (Unnamed, NaN, '')
        df = (
            df.dropna(how='all')
                .loc[:,lambda d: ~d.columns.str.startswith('Unnamed:') & d.columns.notna() & (d.columns != '')]  #.copy()
                .reset_index(drop=True)
        )

        status = status + str(len(df)) + ' data rows and ' + str( len(df.columns)) + ' columns.'
        style_cell_conditional = [{'if': {'column_id': col}, 'maxWidth': '18.75rem', 'minWidth': '6.25rem', 'width': 'auto'} for col in df.columns]
        datatable_data = df.to_dict('records') if (new_df or trigger == 'add-row-column-button') else dash.no_update
        datatable_columns = [{'name': col, 'id': col, 'editable': True, 'deletable': True, 'renamable': True} for col in df.columns] if (new_df or trigger == 'add-row-column-button') else dash.no_update
        original_columns, numeric_columns, categorical_columns, df_row_variable, df_column_variable = update_dataframe_data(df, None, category_suffix)

        row_consistent_columns, column_consistent_columns = get_consistent_columns(df, categorical_columns, df_row_variable, df_column_variable)
        stored_dataframe = {'data': df.to_json(date_format='iso', orient='split'),
                            'dtypes': df.dtypes.astype(str).to_dict(),
                            'categories': {col: df[col].cat.categories.tolist() for col in df.select_dtypes('category')},
                            'ordered': {col: df[col].cat.ordered for col in df.select_dtypes('category')}}

        categorical_columns_list = [{'label': option, 'value': option} for option in categorical_columns]
        numerical_columns_list = [{'label': option, 'value': option} for option in numeric_columns]
        all_columns_list = [{'label': option, 'value': option} for option in original_columns]
        additional_row_columns_list = [{'label': option, 'value': option} for option in row_consistent_columns]
        additional_column_columns_list = [{'label': option, 'value': option} for option in column_consistent_columns]
        multiple_dataframes_handling_options = [{'label': 'First', 'value': 'First'}, {'label': 'All', 'value': 'All'}]
        multiple_dataframes_id_col_options = all_columns_list.copy()
        if df_row_variable and df_column_variable:
            multiple_dataframes_id_col_options.insert(0, {'label': 'Row/Column', 'value': 'Row/Column'})
        multiple_dataframes_id_col_state, multiple_dataframes_id_col_value = False, dash.no_update
        if graph_selection == 'Heatmap' or graph_selection == 'Pie Charts':
            multiple_dataframes_id_col_state = True
            if any(opt['value'] == 'Row/Column' for opt in multiple_dataframes_id_col_options):
                multiple_dataframes_id_col_value = 'Row/Column'
        elif trigger == 'test-data':
            if any(opt['value'] == 'Row/Column' for opt in multiple_dataframes_id_col_options):
                multiple_dataframes_id_col_value = 'Row/Column'

        return (
            # data
            stored_dataframe, f'Current file: {load_filename}' if new_df else dash.no_update, original_columns, df_row_variable, df_column_variable, numeric_columns, categorical_columns,
            # parallel
            str(uuid.uuid4()), all_columns_list, [] if new_df else dash.no_update,
            str(uuid.uuid4()), numerical_columns_list, next((opt['value'] for opt in numerical_columns_list if opt['value'].lower().startswith(('yield', 'product'))), None) if new_df else dash.no_update,
            # scatter
            str(uuid.uuid4()), all_columns_list, next((opt['value'] for opt in all_columns_list if opt['value'].lower() == 'column'), all_columns_list[0]['value']) if (new_df or trigger == 'add-row-column-button') else dash.no_update,
            str(uuid.uuid4()), all_columns_list, [] if new_df else dash.no_update,
            str(uuid.uuid4()), all_columns_list, next((opt['value'] for opt in all_columns_list if opt['value'].lower() == 'row'), all_columns_list[0]['value']) if (new_df or trigger == 'add-row-column-button') else dash.no_update, \
            str(uuid.uuid4()), all_columns_list, None if new_df else dash.no_update,
            False if not datatable_modified else dash.no_update,
            str(uuid.uuid4()), all_columns_list, next((opt['value'] for opt in all_columns_list if opt['value'].lower().startswith(('yield', 'product'))), None) if new_df else dash.no_update,
            str(uuid.uuid4()), all_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), all_columns_list, next((opt['value'] for opt in all_columns_list if opt['value'].lower().startswith(('yield', 'product'))), None) if new_df else dash.no_update,
            # heatmap
            str(uuid.uuid4()), all_columns_list, next((opt['value'] for opt in all_columns_list if opt['value'].lower().startswith(('yield', 'product'))), all_columns_list[0]['value']) if new_df else dash.no_update,
            str(uuid.uuid4()), additional_row_columns_list, [] if new_df else dash.no_update,
            str(uuid.uuid4()), additional_column_columns_list, [] if new_df else dash.no_update,
            # piecharts
            False if new_df else dash.no_update,
            str(uuid.uuid4()), numerical_columns_list, [] if new_df else dash.no_update,
            str(uuid.uuid4()), additional_row_columns_list, [] if new_df else dash.no_update,
            str(uuid.uuid4()), additional_column_columns_list, [] if new_df else dash.no_update,
            # bars
            str(uuid.uuid4()), all_columns_list, next((opt['value'] for opt in all_columns_list if opt['value'].lower().startswith(('id'))), None) if new_df else dash.no_update,
            str(uuid.uuid4()), numerical_columns_list, next((opt['value'] for opt in numerical_columns_list if opt['value'].lower().startswith(('yield', 'product'))), None) if new_df else dash.no_update,
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), dash.no_update, 'Group' if new_df else dash.no_update,
            # dumbbell
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), numerical_columns_list, next((opt['value'] for opt in numerical_columns_list if opt['value'].lower().startswith(('yield', 'product'))), None) if new_df else dash.no_update,
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            # common
            str(uuid.uuid4()), categorical_columns_list, None if new_df else dash.no_update,
            str(uuid.uuid4()), multiple_dataframes_handling_options, 'First' if new_df else dash.no_update,
            multiple_dataframes_id_col_options, multiple_dataframes_id_col_value, multiple_dataframes_id_col_state,
            str(uuid.uuid4()), None if new_df else dash.no_update,

            ##df.to_dict('records') if (new_df or trigger == 'add-row-column-button') else dash.no_update,
            ##[{'name': col, 'id': col, 'editable': True, 'deletable': True, 'renamable': True} for col in df.columns] if (new_df or trigger == 'add-row-column-button') else dash.no_update,
            datatable_data, datatable_columns, {'display': 'block'},
            style_cell_conditional, sort_by_output, filter_query_output, [] if new_df else dash.no_update,
            {'display': 'none'} if new_df else dash.no_update, None if new_df else dash.no_update, datatable_modified, datatable_timestamp, None if new_df else dash.no_update, status,
            )

    except Exception as e:
        exception_message = ''
        if trigger:
            if trigger == 'upload-data' or trigger == 'test-data':
                exception_message = 'Exception processing file: '
            elif trigger == 'add-row-column-button':
                exception_message == 'Exception in adding row/column: '
            else: # trigger == 'datatable'
                exception_message == 'Exception in modifying data table: '
        exception_message = exception_message + generate_exception_message(e)
        return [dash.no_update] * (number_of_load_data_outputs - 1) + [exception_message]

number_of_graph_outputs = 5
#<editor-fold desc="**app.callback => Generate graph">
@app.callback(
    Output('graph-container', 'children', allow_duplicate=True),
    Output('graph-container', 'style', allow_duplicate=True),
    Output('raw-figures-store', 'data', allow_duplicate=True),
    Output('dataframe-store', 'data', allow_duplicate=True),
    Output('status-bar', 'children', allow_duplicate=True),
    Input('generate-button', 'n_clicks'),
    Input('datatable-modified-store', 'data'),
    State('dataframe-store', 'data'),
    State('graph-selection', 'value'),
    State('df-row-variable-store', 'data'),
    State('df-column-variable-store', 'data'),
    State({'type': 'listbox', 'subtype': 'parallel-variables'}, 'value'),
    State('parallel-color-variable', 'value'),
    State('scatter-x-variable', 'value'),
    State({'type': 'listbox', 'subtype': 'scatter-x-subplots-variables'}, 'value'),
    State('scatter-y-variable', 'value'),
    State('scatter-z-variable', 'value'),
    State('scatter-surface', 'value'),
    State('scatter-size-variable', 'value'),
    State('scatter-symbol-variable', 'value'),
    State('scatter-color-variable', 'value'),
    State('heatmap-color-variable', 'value'),
    State({'type': 'listbox', 'subtype': 'heatmap-add-row-variable'}, 'value'),
    State({'type': 'listbox', 'subtype': 'heatmap-add-column-variable'}, 'value'),
    State('heatmap-smooth-option', 'value'),
    State({'type': 'listbox', 'subtype': 'piechart-variables'}, 'value'),
    State({'type': 'listbox', 'subtype': 'piechart-add-row-variable'}, 'value'),
    State({'type': 'listbox', 'subtype': 'piechart-add-column-variable'}, 'value'),
    State('piechart-normalization-type', 'value'),
    State('piechart-normalization-value', 'value'),
    State('piechart-donut', 'value'),
    State('piechart-cakeplots', 'value'),
    State('barchart-x-variable', 'value'),
    State({'type': 'listbox', 'subtype': 'barchart-variables'}, 'value'),
    State('barchart-pattern-variable', 'value'),
    State('barchart-group-variables-by', 'value'),
    State('barchart-barmode-option', 'value'),
    State('dumbbell-x-variable', 'value'),
    State('dumbbell-y-variable', 'value'),
    State({'type': 'listbox', 'subtype': 'dumbbell-grouped-variables'}, 'value'),
    State('dumbbell-color-variable', 'value'),
    State('dumbbell-symbol-variable', 'value'),
    State('colorscale', 'value'),
    State('split-by-variable', 'value'),
    State('plate-rows-as-alpha', 'value'),
    State('multiple-dataframes-handling', 'value'),
    State('multiple-dataframes-id-column', 'value'),
    State('multiple-dataframes-reverse', 'value'),
    State('multiple-series-names', 'value'),
    State('colorscale-reverse', 'value'),
    State('graph-title', 'value'),
    State('graph-container', 'style'),
    prevent_initial_call=True
)
#</editor-fold>

def generate_graph(generate_n_clicks, datatable_modified, stored_dataframe, graph_selection, df_row_variable, df_column_variable, parallel_variables, parallel_color_variable, scatter_x_variable, \
                                scatter_x_subplots_variables, scatter_y_variable, scatter_z_variable, scatter_surface, scatter_size_variable, scatter_symbol_variable, scatter_color_variable, \
                                heatmap_color_variable, heatmap_additional_row_variable, heatmap_additional_column_variable, heatmap_smooth, piechart_variables, piechart_additional_row_variable, piechart_additional_column_variable, \
                                piechart_normalization_type, piechart_normalization_value, piechart_donut, piechart_cakeplots, barchart_x_variable, barchart_variables, barchart_pattern_variable, \
                                barchart_group_variables_by, barchart_barmode_option, dumbbell_x_variable, dumbbell_y_variable, dumbbell_grouped_variables, dumbbell_color_variable, dumbbell_symbol_variable, \
                                colorscale, split_by_variable, plate_rows_as_alpha, multiple_dataframes_handling, multiple_dataframes_id_column, multiple_dataframes_reverse, multiple_dataframes_series_names, colorscale_reverse, \
                                graph_title, current_graph_container_style):

    trigger, figs, graphs, grid, grid_columns, graph_container_style, status, dfs_delimiter = None, [], [], [], 1, {'display': 'none'}, '', '|=%' # we want to use something that will not be used
    dfs, number_of_figures, number_of_figure_rows, number_of_figure_columns, viewport_maximum_width, viewport_maximum_height = {}, 0, 0, 0, 80, 90

    try:
        trigger = ctx.triggered_id
        if trigger == 'datatable-modified-store':
            if datatable_modified == False:
                return [dash.no_update] * number_of_graph_outputs # no changes to datatable so just return
            else: # changes to datatable, so redo the graph
                if current_graph_container_style == graph_container_style:  # no graphs currently so nothing to update
                    return [dash.no_update] * number_of_graph_outputs
                df = pd.read_json(io.StringIO(stored_dataframe['data']), orient='split')
                stored_dataframe_output = dash.no_update
        elif trigger == 'generate-button':
            if generate_n_clicks is None or generate_n_clicks == 0 or stored_dataframe is None: # Don't generate anything on initial load
                return [], {'display': 'none'}, dash.no_update, dash.no_update, 'No datafile or options selected.'
            df = pd.read_json(io.StringIO(stored_dataframe['data']), orient='split') # so we reread a fresh copy
            stored_dataframe_output = dash.no_update
        else:
            return [dash.no_update] * number_of_graph_outputs

        if df.empty: # something was updated, but there are no values
            return figs, graph_container_style, stored_dataframe_output, dash.no_update, 'Empty dataframe.'

        # we are reapplying the correct format/types to the df here;
        for col, dtype in stored_dataframe['dtypes'].items():
            if dtype != 'category':
                df[col] = df[col].astype(dtype)
        for col, cats in stored_dataframe['categories'].items():
            df[col] = pd.Categorical(df[col], categories=cats, ordered=stored_dataframe['ordered'][col])

    except Exception as e:
        exception_message = 'Exception in accessing data: ' + generate_exception_message(e)
        return [dash.no_update] * (number_of_graph_outputs - 1) + [exception_message]

    try:
        # With no multiple_id_column, dfs = {'All': data}; with {'Series 1': data, 'Series 2': data}
        mdi_single_df = False
        if multiple_dataframes_id_column is not None:
            dfs = split_dataframe(df, multiple_dataframes_id_column)
            number_of_dfs = len(dfs)
            if multiple_dataframes_reverse:
                dfs = dict(reversed(list(dfs.items())))
            if multiple_dataframes_handling == 'First':
                dfs = {next(iter(dfs)): dfs[next(iter(dfs))]}
                number_of_dfs = 1
                mdi_single_df = True
            elif multiple_dataframes_handling == 'Last':
                dfs = {next(reversed(dfs)): dfs[next(reversed(dfs))]}
                number_of_dfs = 1
                mdi_single_df = True
            if multiple_dataframes_series_names:
                names = multiple_dataframes_series_names.split('|')
                dfs = {name: subdf for name, subdf in zip(names, dfs.values())}
        else:
            number_of_dfs = 1
            dfs = {'All': df.copy()}

        # With split_by variable, then becomes {'All[del]F1': data, 'All[del]F2': data}, {'Series 1[del]F1': data, 'Series 1[del]F2': data, 'Series 1[del]F1': data, 'Series 2[del]F2': data}
        if split_by_variable:
            split_dfs = {}
            for key, subdf in dfs.items():
                grouped = subdf.groupby(split_by_variable, observed=False, sort=False)
                for split_by, sub_subdf in grouped:
                    split_dfs[f'{key}{dfs_delimiter}{split_by}'] = sub_subdf
            dfs = split_dfs

        # Remove any empty dataframes and remove unused categories
        dfs = {k: v for k, v in dfs.items() if not v.empty}
        ##dfs = {k: v.apply(lambda col: col.cat.remove_unused_categories() if col.dtype.name == 'category' else col) for k, v in dfs.items()}

        # check if any of the columns in the dataframes can be reconverted to numerical and if so, do so
        for key, df  in dfs.items():
            for col in df.select_dtypes(['category', 'object']).columns:
                if pd.to_numeric(df[col], errors='coerce').notna().all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if col + category_suffix in df.columns:
                        del df[col + category_suffix] # clean up some

    except Exception as e:
        exception_message = 'Exception in processing data into dfs: ' + generate_exception_message(e)
        return [dash.no_update] * (number_of_graph_outputs - 1) + [exception_message]

    try:
        if colorscale_reverse:
            colorscale = colorscale + '_r'
        if (graph_selection == 'Parallel Coordinates'):
            if not parallel_variables or not parallel_color_variable:
                status = 'Need to select parallel variables and color variable.' if trigger == 'generate-button' else ''
            else:
                figs = generate_parallel_coordinates_graph(dfs, ensure_list(parallel_variables), parallel_color_variable, colorscale, split_by_variable, plate_rows_as_alpha, multiple_dataframes_id_column, \
                                                                             number_of_dfs, mdi_single_df, graph_title, category_suffix, plate_variables_columns)
                status = 'Generated parallel coordinates graph.'
        elif (graph_selection == 'Scatter'):
            if not scatter_y_variable or (not scatter_x_variable and not scatter_x_subplots_variables):
                status = 'Need to select x and y variables.' if trigger == 'generate-button' else ''
            else:
                figs = generate_scatter_bubble_graph(dfs, scatter_x_variable, endure_list(scatter_x_subplots_variables), scatter_y_variable, scatter_z_variable, scatter_surface, scatter_size_variable, \
                                                     scatter_symbol_variable, scatter_color_variable, colorscale, split_by_variable, plate_rows_as_alpha, multiple_dataframes_id_column, number_of_dfs, \
                                                     mdi_single_df, graph_title, category_suffix, plate_variables_columns)
                status = 'Generated scatter/bubble graph - 3d ignored when doing subplots.' if scatter_x_subplots_variables and scatter_z_variable else 'Generated scatter/bubble graph.'
        elif (graph_selection == 'Heatmap' or graph_selection == 'Pie Charts'):
            if (df_row_variable == None) or (df_column_variable == None):
                status = graph_selection + ' graphs require a valid Row and Column columns in the dataframe/provided data file as they are configured for plate formats.' if trigger == 'generate-button' else ''
            elif isinstance(next(iter(dfs.values()))[df_column_variable].dtype, CategoricalDtype): # we can check the only one since types were assigned on the df before splitting into dfs
                status = 'Dataframe type for Column must be numerical within all dataframes for ' + graph_selection + ' graphs.' # because we can't determine column number without knowing total number of columns
            elif isinstance(next(iter(dfs.values()))[df_row_variable].dtype, CategoricalDtype):
                if all(df[df_row_variable].dropna().astype(str).str.fullmatch(r"[A-Za-z]$").all() for df in dfs.values()) == False:
                    status = 'Dataframe column type for Row must either be numerical or single character categorical within all dataframes for ' + graph_selection + ' graphs.'
                else: # create a column to convert the single characters to numeric row equivalents - we do this for consistency within the other functions
                    for name, df in dfs.items():
                        df['Row_numeric'] = (df[df_row_variable].astype(str).str.upper().map(lambda x: ord(x) - ord('A') + 1 if re.fullmatch(r"[A-Z]", x) else pd.NA))
                        df_row_variable = 'Row_numeric'
            if (graph_selection == 'Heatmap'):
                if not heatmap_color_variable:
                    status = 'Need to select heatmap color variable.' if trigger == 'generate-button' else ''
                else:
                    figs, number_of_figure_rows, number_of_figure_columns = generate_heatmap_graph(dfs, df_row_variable, df_column_variable, heatmap_color_variable, ensure_list(heatmap_additional_row_variable), \
                                                                                           ensure_list(heatmap_additional_column_variable), heatmap_smooth, colorscale, split_by_variable, plate_rows_as_alpha, \
                                                                                           multiple_dataframes_id_column, number_of_dfs, mdi_single_df, graph_title, category_suffix, plate_variables_columns)
                    status = 'Generated heatmap graph.'
            else:  # so Pie Charts
                if not piechart_variables:
                    status = 'Need to select pie chart variables.' if trigger == 'generate-button' else ''
                else:
                    normalization_map = {'Normalize to 100%': 0, 'Values are %': 1, 'Normalize to value': 2}
                    piechart_normalization_index = normalization_map.get(piechart_normalization_type, -1)
                    valid_normalization_value, normalization_value = True, None
                    if (piechart_normalization_index == -1):
                        status = 'Normalization method may not be blank.' if trigger == 'generate-button' else ''
                        valid_normalization_value = False
                    elif (piechart_normalization_index == 2):
                        if piechart_normalization_value is not None:
                            try:
                                normalization_value = float(piechart_normalization_value)
                                if (normalization_value <= 0):
                                    status = 'Normalization value is <=0, positive value is required.' if trigger == 'generate-button' else ''
                                    valid_normalization_value = False
                            except(ValueError, TypeError):
                                status = 'Normalization value entered cannot be converted to float.' if trigger == 'generate-button' else ''
                                valid_normalization_value = False
                        else:
                            status = 'Normalization value may not be null if Normalize to value is selected.' if trigger == 'generate-button' else ''
                            valid_normalization_value = False

                    if valid_normalization_value:
                        status = 'Generated pie charts.'
                        figs, number_of_figure_rows, number_of_figure_columns = generate_piecharts_graph(dfs, df_row_variable, df_column_variable, ensure_list(piechart_variables), \
                                                        ensure_list(piechart_additional_row_variable), ensure_list(piechart_additional_column_variable), piechart_normalization_index, \
                                                        normalization_value, piechart_donut, piechart_cakeplots, colorscale, split_by_variable, plate_rows_as_alpha, multiple_dataframes_id_column, \
                                                        number_of_dfs, multiple_dataframes_reverse, mdi_single_df, graph_title, category_suffix, plate_variables_columns)
        elif (graph_selection == 'Bar Chart'):
            if not barchart_x_variable or not barchart_variables:
                status = 'Need to select x and y variables.' if trigger == 'generate-button' else ''
            else:
                figs = generate_barchart_graph(dfs, barchart_x_variable, ensure_list(barchart_variables), barchart_pattern_variable, barchart_group_variables_by, barchart_barmode_option, \
                                                colorscale, split_by_variable, plate_rows_as_alpha, multiple_dataframes_id_column, multiple_dataframes_reverse, number_of_dfs, mdi_single_df, \
                                                graph_title, category_suffix, plate_variables_columns)
                status = 'Generated bar chart graph.'
        elif (graph_selection == 'Dumbbell Treillis'):
            if not dumbbell_x_variable or not dumbbell_y_variable or not (dumbbell_grouped_variables or dumbbell_color_variable or dumbbell_symbol_variable):
                status = 'Need to select x, y, and grouped over variables for dumbbell treillis plot.' if trigger == 'generate-button' else ''
            else:
                figs = generate_dumbbell_graph(dfs, dumbbell_x_variable, dumbbell_y_variable, dumbbell_color_variable, dumbbell_symbol_variable, ensure_list(dumbbell_grouped_variables), colorscale, \
                                                                 split_by_variable, plate_rows_as_alpha, multiple_dataframes_id_column, number_of_dfs, mdi_single_df, graph_title, category_suffix, plate_variables_columns)
                status = 'Generated dumbbell treillis graph.'
    except Exception as e:
        exception_message = 'Exception in generating ' + graph_selection + ' graph: ' + generate_exception_message(e)
        return [dash.no_update] * (number_of_graph_outputs - 1) + [exception_message]

    try:
        if figs:
            # Note that in Python, the if <value> does not check for only None, it also checks for 'truthiness', and 0 is considered falsy.
            number_of_figures = len(figs)
            grid_columns = 1 if number_of_figures == 1 or (graph_selection == 'Scatter' and (scatter_x_subplots_variables and len(scatter_x_subplots_variables) > 1)) else 2
            graph_style = {'width': '100%', 'height': '100%'}
            html_div_style = {'height': '90vh', 'width': f'{80/grid_columns}vw', 'margin': '0 auto'}

            graphs = [
                html.Div(
                    dcc.Graph(
                       figure=fig,
                        config={'responsive': True},
                        style=graph_style,
                        className='graph-container',
                        clear_on_unhover=True,
                    ),
                    style=html_div_style,
                )
                for fig in figs
            ]

            graph_container_style = {'display': 'block'}
            grid_col_width = bootstrap_cols/grid_columns
            for i in range(0, len(graphs), grid_columns):
                row_children = [dbc.Col(graphs[j], width=grid_col_width) for j in range(i, min(i + grid_columns, len(graphs)))]
                row = dbc.Row(row_children, className='mb-3 g-1', style={'max-height': '90vh', 'overflow': 'hidden'})
                grid.append(row)

        return grid, graph_container_style, graphs, stored_dataframe_output, status

    except Exception as e:
        exception_message = 'Exception in exporting returned figures to Dash grid: ' + generate_exception_message(e)
        return [dash.no_update] * (number_of_graph_outputs - 1) + [exception_message]

#<editor-fold desc="**app.callback => Handle download">
@app.callback(
    Output('download-zip', 'data'),
    Output('status-bar', 'children', allow_duplicate=True),
    Input('download-button', 'n_clicks'),
    State('raw-figures-store', 'data'),
    State('datatable', 'data'),
    State('datatable-modified-store', 'data'),
    prevent_initial_call=True,
)
#</editor-fold>

def download_graph_to_html(n_clicks_download, graphs, table_data, datatable_modified):

    if n_clicks_download is None or n_clicks_download == 0 or not graphs:
        return dash.no_update, dash.no_update

    try:
        current_datetime = datetime.now()
        filename_suffix = ('0' + str(current_datetime.month) if len(str(current_datetime.month)) == 1 else str(current_datetime.month)) + \
                          ('0' + str(current_datetime.day) if len(str(current_datetime.day)) == 1 else str(current_datetime.day)) + (str(current_datetime.year))

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            for i, graph in enumerate(graphs):
                fig = graph['props']['children']['props']['figure']
                html_str = pio.to_html(fig, full_html=True)
                img_bytes = pio.to_image(fig, format='png')
                zf.writestr(f'{filename_suffix}_figure_{i+1}.html', html_str)
                zf.writestr(f'{filename_suffix}_figure_{i+1}.png', img_bytes)
            if datatable_modified:
                table_df = pd.DataFrame(table_data)
                csv_datatable = table_df.to_csv(index=False)
                zf.writestr(filename_suffix + '_datatable.csv', csv_datatable.encode('utf-8'))

        buffer.seek(0)
        return dcc.send_bytes(lambda x: x.write(buffer.getvalue()), filename=filename_suffix + '_data.zip'), 'Downloaded data as ' + filename_suffix + '_data.zip'

    except Exception as e:
        exception_message = 'Exception in downloading data: ' + generate_exception_message(e)
        return dash.no_update, exception_message

# Run the app - this is for local
if __name__ == '__main__':
    app.run(port=8050, debug=True)
