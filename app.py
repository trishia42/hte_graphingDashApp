import pandas as pd
#from pandas.api.types import CategoricalDtype, is_numeric_dtype
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH, dash_table, no_update
import dash_bootstrap_components as dbc
import base64, io, uuid
#from plotly.graph_objects import Figure
import plotly.graph_objects as go
from miscellaneousFunctions import *
from graphingFunctions import *

# Component generation and UI functions
def generate_dropdown_control(id_, options, label, label_style=None, value=None, multi=False, listbox=False, size=10):
    return dbc.Form([
        dbc.Label(label, html_for=id_, style=label_style),
        dcc.Dropdown(
            id={'type': 'listbox', 'subtype': id_} if listbox else id_,
            options=[{'label': i, 'value': i} for i in options] if options else [],
            value=value,
            multi=multi,
            style={"width": "100%", "marginBottom": "0.25rem"},
        )
    ])

def draw_colorscale_preview(colorscale_name, reverse_option):
    if reverse_option:
        colorscale_name = colorscale_name + '_r'
    z = [[i for i in range(100)]]
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale=colorscale_name, showscale=False
    ))
    fig.update_layout(
        #height=60, width=300,
        margin=dict(l=2, r=1, t=0, b=12), xaxis=dict(showticklabels=False, showgrid=False, zeroline=False), yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    fig.add_shape(
        type="rect", x0 = 0, y0=0, x1=1, y1=1, xref="paper", yref="paper", line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)"  # Make it transparent (or choose a background color)
    )

    return dcc.Graph(id='color-scale-preview', figure=fig, config={'displayModeBar': False, 'responsive': True}, style={'width': '95%', 'height':'40px'})

color_scale_options = ['Plotly3', 'Viridis', 'Inferno', 'Magma', 'Plasma', 'Turbo', 'Bluered', 'Electric', 'Hot', 'Jet',
                       'Rainbow', 'Thermal', 'Haline', 'Solar', 'Ice', 'Deep', 'Dense', 'Matter', 'Speed', 'AgSunset',
                       'SunsetDark', 'Aggrnyl', 'Puor', 'RdBu', 'Spectral', 'Balance', 'Delta', 'Curl', 'TealRose', 'Portland']
category_suffix, columns_to_skip_in_UI, datatable_modified = '_encoded', ['row', 'column', 'position', 'index', 'well', 'well index', 'id'], False
bootstrap_cols = 12
sidebar_width = (1/6)*bootstrap_cols # in bootstrap definition
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server=app.server # for render deploy - this is for render hosting
app.title = "Data Visualization Dashboard"

# Actual app layout
app.layout = dbc.Container([
    dcc.Store(id="stored-dataframe"),
    dcc.Store(id='original-cols-store'),
    dcc.Store(id='row-variable-store'),
    dcc.Store(id='column-variable-store'),
    dcc.Store(id='numeric-cols-store'),
    dcc.Store(id='categorical-cols-store'),
    dcc.Store(id='encoded-categorical-cols-store'),
    dcc.Store(id='datatable-modified-store'),
    dbc.Row([
        dbc.Col([
            generate_dropdown_control('graph-type', ["Parallel Coordinates", "Scatter", "Heatmap", "Pie Charts"], "Type of Graph", label_style={"fontSize": "16px", "fontWeight": "bold", "padding": "6px 0px 0px 0px"}, value="Parallel Coordinates"),
            html.Div(id="parallel-options", style={"display": "block"}, children=[
                html.H6("Parallel Coordinates Options", style={"fontSize": "16px", "fontWeight": "bold", 'marginTop':'1rem'}),
                html.Div(id='parallel-vars-wrapper', children=[
                    generate_dropdown_control('parallel-vars', None, 'Variables', None, None, True, True)
                ], style={"marginBottom": "1rem"}),
                html.Div(id='parallel-color-wrapper', children=[
                    generate_dropdown_control('parallel-color', None, "Color Variable", value=None)
                ]),
            ]),
            html.Div(id="scatter-options", style={"display": "none"}, children=[
                html.H6("Scatter Plot Options", style={"fontSize": "16px", "fontWeight": "bold", 'marginTop':'1rem'}),
                html.Div(id='scatter-x-wrapper', children=[
                    generate_dropdown_control('scatter-x', None, "X Axis", None),
                ]),
                html.Div(id='scatter-y-wrapper', children=[
                    generate_dropdown_control('scatter-y', None, "Y Axis", value=None),
                ]),
                html.Div(id='scatter-size-wrapper', children=[
                    generate_dropdown_control('scatter-size', None, "Size", value=None),
                ]),
                html.Div(id='scatter-symbol-wrapper', children=[
                    generate_dropdown_control('scatter-symbol', None, "Symbol", value=None),
                ]),
                html.Div(id='scatter-color-wrapper', children=[
                    generate_dropdown_control('scatter-color', None, "Color", value=None),
                ]),
            ]),
            html.Div(id="heatmap-options", style={"display": "none"}, children=[
                html.H6("Heatmap Options", style={"fontSize": "16px", "fontWeight": "bold", 'marginTop':'1rem'}),
                html.Div(id='heatmap-color-wrapper', children=[
                    generate_dropdown_control('heatmap-color', None, "Color Variable"),
                ]),
                html.Div(id='heatmap-add-row-wrapper', children=[
                    generate_dropdown_control('heatmap-add-row', None, "Additional Row Variable", None, None, True, True),
                ]),
                html.Div(id='heatmap-add-col-wrapper', children=[
                    generate_dropdown_control('heatmap-add-col', None, "Additional Column Variable", None, None, True, True),
                ]),
                generate_dropdown_control('heatmap-smooth', ['False', 'best', 'fast'], "Smooth", value='False')
            ]),
            html.Div(id="piechart-options", style={"display": "none"}, children=[
                html.H6("Pie Chart Options", style={"fontSize": "16px", "fontWeight": "bold", 'marginTop':'1rem'}),
                html.Div(id='piechart-vars-wrapper', children=[
                    generate_dropdown_control('piechart-vars', None, "Variables", None, None, True, True),
                ]),
                html.Div(id='piechart-add-row-wrapper', children=[
                    generate_dropdown_control('piechart-add-row', None, "Additional Row Variable", None, None, True, True)
                ]),
                html.Div(id='piechart-add-col-wrapper', children=[
                    generate_dropdown_control('piechart-add-col', None, "Additional Column Variable", None, None, True, True)
                ]),
                generate_dropdown_control('piechart-norm', ['Normalize to 100%', 'Values are %', 'Normalize to value'], "Normalization", value='Normalize to 100%'),
                dbc.Label("Value (if applicable):"),
                dbc.Input(id='piechart-norm-value', type='text', placeholder="Enter value"),
            ]),
            html.Hr(style={"marginBottom": "0.25rem"}),
            html.Div([
                dbc.Row([
                    dbc.Col(
                        generate_dropdown_control('color-scale', color_scale_options, "Color Scale", value=color_scale_options[0]), width=9, style={"paddingRight": "0.1rem"}
                    ),
                    dbc.Col(
                        dbc.Checkbox(id='color-reverse-scale', label='_r', value=False, style={'marginTop': '50px', 'marginRight': '0px', 'marginLeft': '0px'},), width=3, style={"display": "flex", "alignItems": "flex-end", "justifyContent": "flex-start", "paddingRight": 0, "paddigLeft":0}
                    ),
                ], align="center", style={'marginBottom':'0.5rem'}), # added
                draw_colorscale_preview(color_scale_options[0], False)
            ], style={'width': '100%', 'marginBottom':'0rem'} # changed from 1rem to 0.5rem
            ),
            dbc.Label("Graph Title"),
            dbc.Input(id='graph-title', type='text', placeholder="Title", style={"marginBottom": "1rem"}),
            dbc.Button("Generate", id="generate-button", color="success", style={'display':'flex', "width": "120px", 'height':'35px', 'justifyContent':'center', 'alignItems':'center', 'marginTop':'1rem', 'marginLeft':'0.5rem'}), # changed button config
            #html.Div(
            #    dbc.Button("Generate", id="generate-button", color="success", style={'display':'flex', "width": "120px", 'height':'35px', 'justifyContent':'center', 'alignItems':'center', 'marginTop':'1rem'}), # changed button config
            #    #style={'textAlign':'right'}
            #    style={"display": "flex", "justifyContent": "flex-end", "marginTop": "0.5rem"}
            #),
            html.Div([
                dcc.Upload(id='upload-data', children=dbc.Button('Upload File', color="primary", style={'display':'flex', "width": "120px", 'height':'35px', 'justifyContent':'center', 'alignItems':'center'}), multiple=False),
            ], style={"marginTop": "0.5rem", 'marginLeft':'0.5rem'}), # changed from 1rem
            #html.Div(id='upload-output', style={'padding': "10px"}), # moved down
            html.Div(
                dbc.Button("Download", id="download-button", color="secondary",  style={'display':'flex', "width": "120px", 'height':'35px', 'justifyContent':'center', 'alignItems':'center', 'marginTop':'0.5rem', 'marginLeft':'0.5rem'}),
                style={'textAlign':'right'},
            ),
            dcc.Download(id="download-graph-html"),
            dcc.Download(id="download-csv-datatable"),
            html.Div([
                dbc.Button("Test Data", id="test-data", color="info", style={'display':'flex', "width": "120px", 'height':'35px', 'justifyContent':'center', 'alignItems':'center'}),
            ], style={"marginTop": "0.5rem", 'marginLeft':'0.5rem'}), # changed from 1 rem
            html.Div(id='upload-output', style={'padding': "10px", 'marginLeft':'0.5rem'}),
        #], width=sidebar_width, style={"borderRight": "1px solid #ddd", "overflow": "visible", "position": "relative", "zIndex": "1"}),
        ], width=sidebar_width, style={"borderRight": "1px solid #ddd", "overflowY": "scroll", 'height':'100vh', 'scrollbarwidth':'none', 'msOverflowStyle':'none'}),

        dbc.Col([
            html.Div(id="app-note", children="App will spin down after 15 minutes of inactivity, requiring another 50-60 seconds to reactivate.  Please notify me of any issues you encounter.", \
                     style={"backgroundColor": "#f8f9fa","padding": "2px 2px", "marginBottom": "5px", "marginTop": "5px", "fontSize": "14px", "width": "100%",'fontStyle':'italic'}),
            html.Div(id="status-bar", children="Status", style={"backgroundColor": "#f8f9fa", "border": "1px solid #ccc","padding": "8px 12px", "marginBottom": "5px", "marginTop": "5px", \
                     "fontWeight": "bold", "fontSize": "16px", "width": "100%",}),
            html.Div([
                dcc.Loading(
                    dcc.Graph(
                        id="graph-output",
                        config={"responsive": True},
                        style={"height": "90vh", "display": "none"},
                        className="graph-container",
                        responsive=True,
                        clear_on_unhover=True
                    )
                ),
                html.Hr(),
                html.Div(
                    id='data-table-container', style={'display': 'none'}, children=[
                        html.H5("Current Data Table"),
                        dash_table.DataTable(id='data-table', data=[], columns=[], page_size=10, style_table={'overflowX': 'auto', 'border': '1px solid lightgrey'}, editable=True,
                                     style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid lightgrey'},
                                     style_header={'fontWeight': 'bold', 'border': '1px solid lightgrey'}, style_cell_conditional=[],
                                     ),
            ]),
        ])
        ], width=(bootstrap_cols - sidebar_width), style={"overflowY": "scroll", 'height':'100vh', 'scrollbarwidth':'none', 'msOverflowStyle':'none'})
    ])
], fluid=True)

# Panels visibility based on graph_type
@app.callback(
    Output("parallel-options", "style"),
    Output("scatter-options", "style"),
    Output("heatmap-options", "style"),
    Output("piechart-options", "style"),
    Input("graph-type", "value")
)
def toggle_panels_callback(graph_type):
    def show(name):
        return {"display": "block"} if graph_type == name else {"display": "none"}
    return (
        show("Parallel Coordinates"),
        show("Scatter"),
        show("Heatmap"),
        show("Pie Charts")
    )

# Color scale preview
@app.callback(
    Output('color-scale-preview', 'figure'),
    Input('color-scale', 'value'),
    Input('color-reverse-scale', 'value')
)
def update_colorscale_preview(colorscale_name, reverse_option):
    return draw_colorscale_preview(colorscale_name, reverse_option).figure

number_of_dataframe_outputs = 54
# Handle uploads/test data
@app.callback(
    Output('stored-dataframe', 'data', allow_duplicate=True),
    Output('upload-output', 'children', allow_duplicate=True),
    Output('original-cols-store', 'data', allow_duplicate=True),
    Output('row-variable-store', 'data', allow_duplicate=True),
    Output('column-variable-store', 'data', allow_duplicate=True),
    Output('numeric-cols-store', 'data', allow_duplicate=True),
    Output('categorical-cols-store', 'data', allow_duplicate=True),
    Output('encoded-categorical-cols-store', 'data', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'parallel-vars'}, 'options', allow_duplicate=True),
    Output('parallel-vars-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'parallel-vars'}, 'value', allow_duplicate=True),
    Output('parallel-color', 'options', allow_duplicate=True),
    Output('parallel-color-wrapper', 'key', allow_duplicate=True),
    Output('parallel-color', 'value', allow_duplicate=True),
    Output('scatter-x', 'options', allow_duplicate=True),
    Output('scatter-x-wrapper', 'key', allow_duplicate=True),
    Output('scatter-x', 'value', allow_duplicate=True),
    Output('scatter-y', 'options', allow_duplicate=True),
    Output('scatter-y-wrapper', 'key', allow_duplicate=True),
    Output('scatter-y', 'value', allow_duplicate=True),
    Output('scatter-size', 'options', allow_duplicate=True),
    Output('scatter-size-wrapper', 'key', allow_duplicate=True),
    Output('scatter-size', 'value', allow_duplicate=True),
    Output('scatter-symbol', 'options', allow_duplicate=True),
    Output('scatter-symbol-wrapper', 'key', allow_duplicate=True),
    Output('scatter-symbol', 'value', allow_duplicate=True),
    Output('scatter-color', 'options', allow_duplicate=True),
    Output('scatter-color-wrapper', 'key', allow_duplicate=True),
    Output('scatter-color', 'value', allow_duplicate=True),
    Output('heatmap-color', 'options', allow_duplicate=True),
    Output('heatmap-color-wrapper', 'key', allow_duplicate=True),
    Output('heatmap-color', 'value', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-row'}, 'options', allow_duplicate=True),
    Output('heatmap-add-row-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-row'}, 'value', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-col'}, 'options', allow_duplicate=True),
    Output('heatmap-add-col-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'heatmap-add-col'}, 'value', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-vars'}, 'options', allow_duplicate=True),
    Output('piechart-vars-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-vars'}, 'value', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-row'}, 'options', allow_duplicate=True),
    Output('piechart-add-row-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-row'}, 'value', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-col'}, 'options', allow_duplicate=True),
    Output('piechart-add-col-wrapper', 'key', allow_duplicate=True),
    Output({'type': 'listbox', 'subtype': 'piechart-add-col'}, 'value', allow_duplicate=True),
    Output('data-table', 'data', allow_duplicate=True),
    Output('data-table', 'columns', allow_duplicate=True),
    Output('data-table-container', 'style', allow_duplicate=True),
    Output('data-table', 'style_cell_conditional', allow_duplicate=True),
    Output('graph-output', 'figure', allow_duplicate=True),
    Output('graph-output', 'style', allow_duplicate=True),
    Output('status-bar', 'children', allow_duplicate=True),
    Output('datatable-modified-store', 'data', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Input('test-data', 'n_clicks'),
    prevent_initial_call=True
)

def load_data(contents, filename, n_clicks):
    trigger = ctx.triggered_id
    if trigger == 'upload-data':
        if contents is None:
            return [dash.no_update] * number_of_dataframe_outputs
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data_to_load = io.BytesIO(decoded)
    elif trigger == 'test-data':
        if n_clicks is None or n_clicks == 0:
            return [dash.no_update] * number_of_dataframe_outputs
        data_to_load, filename = "testData.xlsx", "testData.xlsx"
    else:
        return [dash.no_update] * number_of_dataframe_outputs

    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(data_to_load, na_values=[""], keep_default_na=False)
        elif filename.endswith(".csv"):
            df = pd.read_csv(data_to_load, na_values=[""], keep_default_na=False)
        else:
            return [dash.no_update] * (number_of_dataframe_outputs - 1) + ["Only .xlsx/xls and .csv files are currently supported."]
    except Exception as e:
        return [dash.no_update] * (number_of_dataframe_outputs - 1) + [f"❌ Error processing file: {e}"]

    datatable_modified = False # on a new load

    try:
        # Drop empty lines (removes blank rows) and columns without headers (Unnamed, NaN, '')
        df = df.dropna(how='all').loc[:,lambda d: ~d.columns.str.startswith('Unnamed:') & d.columns.notna() & (d.columns != '')]  #.copy()
        stored_dataframe = df.to_json(date_format='iso', orient='split')
        data_data = df.to_dict('records')
        data_columns = [{'name': col, 'id': col, 'editable': True} for col in df.columns]
        style_cell_conditional = [{'if': {'column_id': col}, 'maxWidth': '300px', 'minWidth': '100px', 'width': 'auto',} for col in df.columns]
        original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_variable, column_variable = update_dataframe_data(df, None, category_suffix)

        parallel_vars = [c for c in categorical_cols + numeric_cols if c.lower() not in columns_to_skip_in_UI]
        parallel_vars_listbox_options = [{'label': option, 'value': option} for option in parallel_vars]
        parallel_color_options = [{'label': c, 'value': c} for c in numeric_cols if c.lower() not in columns_to_skip_in_UI]
        scatter_xy_options = [{'label': option, 'value': option} for option in [c for c in original_cols]]
        scatter_options = [{'label': c, 'value': c} for c in original_cols if c.lower() not in columns_to_skip_in_UI]
        heatmap_color_options = [{'label': c, 'value': c} for c in numeric_cols + categorical_cols if c.lower() not in columns_to_skip_in_UI]
        piechart_vars = [c for c in numeric_cols if c.lower() not in columns_to_skip_in_UI]
        piechart_vars_listbox_options = [{'label': option, 'value': option} for option in piechart_vars]
        row_consistent_columns, column_consistent_columns = get_consistent_columns(df, categorical_cols, row_variable, column_variable)
        add_row_variables = [{'label': c, 'value': c} for c in row_consistent_columns if c.lower() not in columns_to_skip_in_UI]
        add_column_variables = [{'label': c, 'value': c} for c in column_consistent_columns if c.lower() not in columns_to_skip_in_UI]

        # Clear/hide graph
        empty_fig = go.Figure()
        graph_style = {'display': 'none', 'height': '90vh'}

        return (
            stored_dataframe, f"Current file: {filename}", original_cols, row_variable, column_variable, numeric_cols, categorical_cols, encoded_categorical_cols,

            parallel_vars_listbox_options, str(uuid.uuid4()), [], parallel_color_options, str(uuid.uuid4()), \
            next((opt['value'] for opt in parallel_color_options if opt['value'].lower().startswith('yield')), 'None'),

            scatter_xy_options, str(uuid.uuid4()), next((opt['value'] for opt in scatter_xy_options if opt['value'].lower() == 'column'), scatter_xy_options[0]['value']), \
            scatter_xy_options, str(uuid.uuid4()), next((opt['value'] for opt in scatter_xy_options if opt['value'].lower() == 'row'), scatter_xy_options[0]['value']), \
            scatter_options, str(uuid.uuid4()), next((opt['value'] for opt in scatter_options if opt['value'].lower().startswith('yield')), 'None'), scatter_options, \
            str(uuid.uuid4()), 'None', scatter_options, str(uuid.uuid4()), next((opt['value'] for opt in scatter_options if opt['value'].lower().startswith('yield')), 'None'),

            heatmap_color_options, str(uuid.uuid4()), next((opt['value'] for opt in heatmap_color_options if opt['value'].lower().startswith('yield')), heatmap_color_options[0]['value']), \
            add_row_variables, str(uuid.uuid4()), [], add_column_variables, str(uuid.uuid4()), [], piechart_vars_listbox_options, str(uuid.uuid4()), [], add_row_variables, str(uuid.uuid4()), \
            [], add_column_variables, str(uuid.uuid4()), [],

            data_data, data_columns, {'display': 'block'}, style_cell_conditional, empty_fig, graph_style, "File successfully loaded.",
            False #datatable-modified

            )
    except Exception as e:
        return [dash.no_update] * (number_of_dataframe_outputs - 1) + (f"❌ Error processing file: {e}")

number_of_graph_outputs = 4
# Allow changes to dataframe from datatable/Generate graph
@app.callback(
    Output('graph-output', 'figure', allow_duplicate=True),
    Output('graph-output', 'style', allow_duplicate=True),
    Output('status-bar', 'children', allow_duplicate=True),
    Output('stored-dataframe', 'data', allow_duplicate=True),
    Output('datatable-modified-store', 'data', allow_duplicate=True),
    Input('generate-button', 'n_clicks'),
    Input('data-table', 'data_timestamp'),
    State('stored-dataframe', 'data'),
    State('data-table', 'data'),
    State('data-table', 'columns'),
    State('graph-type', 'value'),
    State('original-cols-store', 'data'),
    State('row-variable-store', 'data'),
    State('column-variable-store', 'data'),
    State('numeric-cols-store', 'data'),
    State('categorical-cols-store', 'data'),
    State('encoded-categorical-cols-store', 'data'),
    State({'type': 'listbox', 'subtype': 'parallel-vars'}, 'value'),
    State('parallel-color', 'value'),
    State('scatter-x', 'value'),
    State('scatter-y', 'value'),
    State('scatter-size', 'value'),
    State('scatter-symbol', 'value'),
    State('scatter-color', 'value'),
    State('heatmap-color', 'value'),
    State({'type': 'listbox', 'subtype': 'heatmap-add-row'}, 'value'),
    State({'type': 'listbox', 'subtype': 'heatmap-add-col'}, 'value'),
    State('heatmap-smooth', 'value'),
    State({'type': 'listbox', 'subtype': 'piechart-vars'}, 'value'),
    State({'type': 'listbox', 'subtype': 'piechart-add-row'}, 'value'),
    State({'type': 'listbox', 'subtype': 'piechart-add-col'}, 'value'),
    State('piechart-norm', 'value'),
    State('piechart-norm-value', 'value'),
    State('color-scale', 'value'),
    State('graph-title', 'value'),
    State('color-reverse-scale', 'value'),
    prevent_initial_call=True
)

def generate_graph(n_clicks, timestamp, stored_data, table_data, columns, graph_type, original_cols, row_variable, column_variable, numeric_cols, categorical_cols,
                                encoded_categorical_cols, parallel_vars, parallel_color, scatter_x, scatter_y, scatter_size,
                                scatter_symbol, scatter_color, heatmap_color, heatmap_add_row, heatmap_add_col, heatmap_smooth,
                                piechart_vars, piechart_add_row, piechart_add_col, piechart_norm, piechart_norm_value,
                                color_scale, graph_title, color_reverse_scale):

    datatable_modified = False

    trigger = ctx.triggered_id
    if trigger == 'data-table':
        if not timestamp or not table_data:
            return [dash.no_update] * number_of_graph_outputs
        df = pd.DataFrame(table_data)[[col['id'] for col in columns]]
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass  # leave as is
        stored_dataframe_output = df.to_json(date_format='iso', orient='split')
        datatable_modified = True # something changed
    elif trigger == 'generate-button':
        if n_clicks is None or n_clicks == 0 or stored_data is None: # Don't gennerate anything on initial load
            return go.Figure(), {'display': 'none'}, "No datafile or options selected.", dash.no_update
        df = pd.read_json(io.StringIO(stored_data), orient='split')
        stored_dataframe_output = dash.no_update
    else:
        return [dash.no_update] * number_of_graph_outputs
    print('drawing')
    print(datatable_modified)
    original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_variable, column_variable = update_dataframe_data(df, None, category_suffix)
    if color_reverse_scale:
        color_scale = color_scale + '_r'
    fig = go.Figure()
    style = {'display': 'none'}
    status = ""
    base_graph_style = {'display': 'block', 'height': '90vh'}  # base style

    if (graph_type == 'Parallel Coordinates'):
        if not parallel_vars or not parallel_color or parallel_color == 'None':
            status = "Need to select parallel variables and color variable."
        else:
            fig, style, status = generate_parallel_coordinates_graph(df, parallel_vars, parallel_color, color_scale, graph_title, category_suffix), \
                {**base_graph_style, 'width': '90%', 'height':'85vh'}, "Generated parallel coordinates graph."
    elif (graph_type == 'Scatter'):
        if not scatter_x or not scatter_y:
            status = "Need to select x and y variables."
        else:
            fig, style, status = generate_scatter_bubble_graph(df, scatter_x, scatter_y, scatter_size, scatter_symbol, scatter_color, color_scale, graph_title, category_suffix), \
                {**base_graph_style, 'width': '90%'}, "Generated scatter/bubble graph."
    elif (graph_type == 'Heatmap' or graph_type == 'Pie Charts'):
        if (row_variable == None) or (column_variable == None):
            status = graph_type + " graphs require a valid 'Row' and 'Column' columns in the dataframe/provided data file."
        elif (graph_type == 'Heatmap'):
            if not heatmap_color:
                status = "Need to select heatmap color variable."
            else:
                fig, style, status = generate_heatmap_graph(df, row_variable, column_variable, heatmap_color, heatmap_add_row, heatmap_add_col, heatmap_smooth, color_scale, graph_title), \
                    {**base_graph_style, 'width': '80%'}, "Generated heatmap graph."
        else:  # so Pie Charts
            if not piechart_vars:
                status = "Need to select pie chart variables."
            else:
                normalization_map = {"Normalize to 100%": 0, "Values are %": 1, "Normalize to value": 2}
                piechart_norm_index = normalization_map.get(piechart_norm, -1)
                valid_norm_value, norm_value = True, None
                if (piechart_norm_index == -1):
                    status = "Normalization method may not be blank."
                    valid_norm_value = False
                elif (piechart_norm_index == 2):
                    if piechart_norm_value:
                        try:
                            norm_value = float(piechart_norm_value)
                            if (norm_value <= 0):
                                status = "Normalization value is <=0, positive value is required."
                                valid_norm_value = False
                                # return go.Figure(), {
                                #    'display': 'none'}, "Normalization value is <=0, positive value is required."
                        except(ValueError, TypeError):
                            status = "Normalization value entered cannot be converted to float."
                            valid_norm_value = False
                    else:
                        status = "Normalization value may not be null if Normalize to value is selected."
                        valid_norm_value = False

                if valid_norm_value:
                    fig, style, status = generate_piecharts_graph(df, row_variable, column_variable, piechart_vars, piechart_add_row, piechart_add_col, piechart_norm_index, norm_value, color_scale, graph_title), \
                        {**base_graph_style, 'width': '80%'}, status

    return fig, style, status, stored_dataframe_output, datatable_modified

# Handle download
@app.callback(
    [Output("download-graph-html", "data"),
     Output("download-csv-datatable", "data")],
    Input("download-button", "n_clicks"),
    State("graph-output", "figure"),
    State("data-table", "data"),
    State("datatable-modified-store", "data"),
    prevent_initial_call=True,
)
def download_graph_to_html(n_clicks, figure, table_data, datatable_modified):
    if not figure:
        return dash.no_update

    fig = go.Figure(figure)
    import plotly.io as pio
    #fig = pio.from_json(figure)
    html_str = pio.to_html(fig, full_html=True)

    if datatable_modified:
        table_df = pd.DataFrame(table_data)
        #csv_buffer = io.StringIO()
        #table_df.to_csv(csv_buffer, index=False)
        csv_datatable = table_df.to_csv(index=False)
        return(
            dict(content=html_str, filename="plotly_graph.html"),
            dict(content=csv_datatable, filename="table_data.csv")
        )
    else:
        return (
            dict(content=html_str, filename="plotly_graph.html"),
            no_update
        )

    #return dict(content=html_str, filename="plotly_graph.html")

# Run the app - this is for local
if __name__ == '__main__':
    app.run(port=8050, debug=True)
