import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH, dash_table
import dash_bootstrap_components as dbc
import base64, io
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from miscellaneousFunctions import convert_categorical, item_exists_in_array, check_row_column_provided, \
    check_additional_row_column_variables, convert_extract_df_columns
from graphingFunctions import generate_parallel_coordinates_graph, generate_scatter_bubble_graph, generate_heatmap_graph, \
    generate_piecharts_graph


def update_dataframe_data(df, category_suffix):
    original_cols = list(df.columns)
    numeric_cols, categorical_cols, encoded_categorical_cols = convert_extract_df_columns(df, None, category_suffix)
    row_var = None
    col_var = None
    if (item_exists_in_array(original_cols, 'row') != -1):
        row_var = 'Row' if 'Row' in original_cols else 'row'
    if (item_exists_in_array(original_cols, 'column') != -1):
        col_var = 'Column' if 'Column' in original_cols else 'column'

    return original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_var, col_var


# Component generation and UI functions
def toggle_option_panels(graph_type):
    def show(name):
        return {"display": "block"} if graph_type == name else {"display": "none"}

    return (
        show("Parallel Coordinates"),
        show("Scatter"),
        show("Heatmap"),
        show("Pie Charts")
    )


def generate_dropdown_control(id_, options, label, label_style=None, value=None, multi=False, listbox=False, size=10):
    return dbc.Form([
        dbc.Label(label, html_for=id_, style=label_style),
        dcc.Dropdown(
            # id=id_,
            id={'type': 'listbox', 'subtype': id_} if listbox else id_,
            options=[{'label': i, 'value': i} for i in options] if options else [],
            value=value,
            multi=multi,
            # className='dropdown-on-top',
            # zindex = 9999,
            # menuPortalTarget='document.body',
            style={"width": "100%", "marginBottom": "1rem",
                   # "height": "auto", #"zIndex": 1, #"minHeight": "100px" if listbox else "auto", #"size": len(options) if listbox else None, ## "zIndex": 9999,
                   # ## "position": "relative", ##"padding": "50px", ##"overflowY": "auto", #"minHeight": "100px" if listbox else "auto",
                   },
        )
    ])


def draw_colorscale_preview(colorscale_name):
    z = [[i for i in range(100)]]
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale=colorscale_name, showscale=False
    ))
    fig.update_layout(
        # height=60, width=300,
        margin=dict(l=2, r=1, t=0, b=12), xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    fig.add_shape(
        type="rect", x0=0, y0=0, x1=1, y1=1, xref="paper", yref="paper", line=dict(color="black", width=1),
        fillcolor="rgba(0,0,0,0)"  # Make it transparent (or choose a background color)
    )

    return dcc.Graph(id='color-scale-preview', figure=fig, config={'displayModeBar': False, 'responsive': True},
                     style={'width': '95%', 'height': '40px'})


color_scale_options = ['Plotly3', 'Viridis', 'Inferno', 'Magma', 'Plasma', 'Turbo', 'Bluered', 'Electric', 'Hot', 'Jet',
                       'Rainbow', 'Thermal', 'Haline', 'Solar', 'Ice', 'Deep', 'Dense', 'Matter', 'Speed', 'AgSunset',
                       'SunsetDark', 'Aggrnyl', 'RdBu']
category_suffix = '_encoded'
columns_to_skip_in_UI = ['row', 'column', 'position', 'index', 'well', 'well index']
bootstrap_cols = 12  # in bootstrap definition
sidebar_width = (1 / 6) * bootstrap_cols

# datafile = "D:\\Desktop\\testData.xlsx"
# df = pd.read_excel(datafile, keep_default_na=False)
# initial_df_json = df.to_json(date_format='iso', orient='split')
# original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_variable, column_variable = update_dataframe_data(df, category_suffix)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
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
    dbc.Row([
        dbc.Col([
            generate_dropdown_control('graph-type', ["Parallel Coordinates", "Scatter", "Heatmap", "Pie Charts"],
                                      "Type of Graph", label_style={"fontSize": "16px", "fontWeight": "bold",
                                                                    "padding": "6px 0px 0px 0px"},
                                      value="Parallel Coordinates"),
            html.Div(id="parallel-options", style={"display": "block"}, children=[
                html.H6("Parallel Coordinates Options", style={"fontSize": "16px", "fontWeight": "bold"}),
                html.Div([
                    generate_dropdown_control('parallel-vars', None, 'Variables', None, None, True, True)
                ], style={"marginBottom": "1rem"}),
                generate_dropdown_control('parallel-color', None, "Color Variable", value=None),
            ]),
            html.Div(id="scatter-options", style={"display": "none"}, children=[
                html.H6("Scatter Plot Options", style={"fontSize": "16px", "fontWeight": "bold"}),
                generate_dropdown_control('bubble-x', None, "X Axis", None),
                generate_dropdown_control('bubble-y', None, "Y Axis", value=None),
                generate_dropdown_control('bubble-size', None, "Size", value=None),
                generate_dropdown_control('bubble-symbol', None, "Symbol", value=None),
                generate_dropdown_control('bubble-color', None, "Color", value=None),
            ]),
            html.Div(id="heatmap-options", style={"display": "none"}, children=[
                html.H6("Heatmap Options", style={"fontSize": "16px", "fontWeight": "bold"}),
                generate_dropdown_control('heatmap-color', None, "Color Variable"),
                generate_dropdown_control('heatmap-add-row', None, "Additional Row Variable", value=None),
                generate_dropdown_control('heatmap-add-col', None, "Additional Column Variable", value=None),
                generate_dropdown_control('heatmap-smooth', ['False', 'best', 'fast'], "Smooth", value='False')
            ]),
            html.Div(id="pie-options", style={"display": "none"}, children=[
                html.H6("Pie Chart Options", style={"fontSize": "16px", "fontWeight": "bold"}),
                generate_dropdown_control('pie-vars', None, "Variables", None, None, True, True),
                generate_dropdown_control('pie-add-row', None, "Additional Row Variable", value=None),
                generate_dropdown_control('pie-add-col', None, "Additional Column Variable", value=None),
                generate_dropdown_control('pie-norm', ['Normalize to 100%', 'Values are %', 'Normalize to value'],
                                          "Normalization", value='Normalize to 100%'),
                dbc.Label("Value (if applicable):"),
                dbc.Input(id='pie-norm-value', type='text', placeholder="Enter value"),
            ]),
            html.Hr(),
            html.Div([
                generate_dropdown_control('color-scale', color_scale_options, "Color Scale",
                                          value=color_scale_options[0]),
                draw_colorscale_preview(color_scale_options[0])
            ], style={'width': '100%', 'marginBottom': '1rem'}
            ),
            dbc.Label("Graph Title"),
            dbc.Input(id='graph-title', type='text', placeholder="Title", style={"marginBottom": "1rem"}),
            dbc.Button("Generate", id="generate-button", color="success", className="mt-2", style={"width": "120px"}),
            html.Div([
                dcc.Upload(id='upload-data',
                           children=dbc.Button('Upload File', color="primary", style={"width": "120px"}),
                           multiple=False),
            ], style={"marginTop": "1rem"}),
            html.Div(id='upload-output', style={'padding': "10px"}),
            dbc.Button("Download", id="download-button", color="secondary", style={"width": "120px"}),
            dcc.Download(id="download-graph-html")
        ], width=sidebar_width,
            style={"borderRight": "1px solid #ddd", "overflow": "visible", "position": "relative", "zIndex": "1"}),
        dbc.Col([
            html.Div(id="status-bar", children="Status",
                     style={"backgroundColor": "#f8f9fa", "border": "1px solid #ccc", "padding": "8px 12px",
                            "marginBottom": "10px", "marginTop": "10px", "fontWeight": "bold", "fontSize": "16px",
                            "width": "100%", }),
            # dcc.Loading(dcc.Graph(id="graph-output", config={"responsive": True}, style={"height": "90vh", "display": "none"}, className="graph-container", responsive=True, clear_on_unhover=True)),
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
                # html.H5("Current Data Table"),
                # dash_table.DataTable(id='data-table', data=df[original_cols].to_dict('records'), columns=[{'name': col, 'id': col} for col in original_cols], page_size=10, style_table={'overflowX': 'auto', 'border': '1px solid lightgrey'},
                #                  style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid lightgrey'}, style_header={'fontWeight': 'bold', 'border': '1px solid lightgrey'},
                #                  style_cell_conditional=[{'if': {'column_id': col}, 'maxWidth': '300px', 'minWidth': '100px', 'width': 'auto',} for col in df.columns],
                # ),
                html.Div(
                    id='data-table-container', style={'display': 'none'}, children=[
                        html.H5("Current Data Table"),
                        dash_table.DataTable(id='data-table', data=[], columns=[], page_size=10,
                                             style_table={'overflowX': 'auto', 'border': '1px solid lightgrey'},
                                             style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal',
                                                         'height': 'auto', 'border': '1px solid lightgrey'},
                                             style_header={'fontWeight': 'bold', 'border': '1px solid lightgrey'},
                                             style_cell_conditional=[],
                                             # style_cell_conditional=[{'if': {'column_id': col}, 'maxWidth': '300px', 'minWidth': '100px', 'width': 'auto', } for col in data.columns],
                                             ),
                    ]),
            ])
        ], width=(bootstrap_cols - sidebar_width))
        # ], style={"width": "50vw"})
    ])
], fluid=True)


# Panels visibility based on graph_type
@app.callback(
    Output("parallel-options", "style"),
    Output("scatter-options", "style"),
    Output("heatmap-options", "style"),
    Output("pie-options", "style"),
    Input("graph-type", "value")
)
def toggle_panels_callback(graph_type):
    return toggle_option_panels(graph_type)


# Color scale preview
@app.callback(
    Output('color-scale-preview', 'figure'),
    Input('color-scale', 'value')
)
def update_colorscale_preview(colorscale_name):
    return draw_colorscale_preview(colorscale_name).figure


# Handle uploads
@app.callback(
    Output('stored-dataframe', 'data'),
    Output('upload-output', 'children'),
    Output('original-cols-store', 'data'),
    Output('row-variable-store', 'data'),
    Output('column-variable-store', 'data'),
    Output('numeric-cols-store', 'data'),
    Output('categorical-cols-store', 'data'),
    Output('encoded-categorical-cols-store', 'data'),

    Output({'type': 'listbox', 'subtype': 'parallel-vars'}, 'options'),
    Output('parallel-color', 'options'),
    Output('parallel-color', 'value'),

    Output('bubble-x', 'options'),
    Output('bubble-x', 'value'),
    Output('bubble-y', 'options'),
    Output('bubble-y', 'value'),
    Output('bubble-size', 'options'),
    Output('bubble-size', 'value'),
    Output('bubble-symbol', 'options'),
    # Output('bubble-symbol', 'value'),
    Output('bubble-color', 'options'),
    Output('bubble-color', 'value'),

    Output('heatmap-color', 'options'),
    Output('heatmap-color', 'value'),
    Output('heatmap-add-row', 'options'),
    Output('heatmap-add-col', 'options'),

    Output({'type': 'listbox', 'subtype': 'pie-vars'}, 'options'),
    Output('pie-add-row', 'options'),
    Output('pie-add-col', 'options'),

    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('data-table-container', 'style'),
    Output('data-table', 'style_cell_conditional'),

    Output('graph-output', 'figure', allow_duplicate=True),
    Output('graph-output', 'style', allow_duplicate=True),

    Output('status-bar', 'children', allow_duplicate=True),

    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    if contents is None:
        return [dash.no_update] * 33  # total number of outputs

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(decoded), na_values=[""], keep_default_na=False)
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(decoded, na_values=[""], keep_default_na=False))
        else:
            return [dash.no_update] * 32 + "Only .xlsx/xls and .csv files are currently supported."

        stored_dataframe = df.to_json(date_format='iso', orient='split')
        data_data = df.to_dict('records')
        data_columns = [{'name': col, 'id': col} for col in df.columns]
        style_cell_conditional = [
            {'if': {'column_id': col}, 'maxWidth': '300px', 'minWidth': '100px', 'width': 'auto', } for col in
            df.columns]

        original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_variable, column_variable = update_dataframe_data(
            df, category_suffix)
        parallel_vars = [c for c in categorical_cols + numeric_cols if c.lower() not in columns_to_skip_in_UI]
        parallel_vars_listbox_options = [{'label': option, 'value': option} for option in parallel_vars]
        parallel_color_options = [{'label': 'None', 'value': 'None'}] + [{'label': c, 'value': c} for c in numeric_cols
                                                                         if c.lower() not in columns_to_skip_in_UI]

        bubble_xy_options = [{'label': option, 'value': option} for option in [c for c in original_cols]]
        bubble_options = [{'label': option, 'value': option} for option in
                          ['None'] + [c for c in original_cols if c.lower() not in columns_to_skip_in_UI]]

        heatmap_color_options = [{'label': option, 'value': option} for option in
                                 [c for c in numeric_cols + categorical_cols if c.lower() not in columns_to_skip_in_UI]]
        heatmap_add_options = [{'label': option, 'value': option} for option in
                               [c for c in ['None'] + categorical_cols if c.lower() not in columns_to_skip_in_UI]]

        pie_vars = [c for c in numeric_cols if c.lower() not in columns_to_skip_in_UI]
        pie_vars_listbox_options = [{'label': option, 'value': option} for option in pie_vars]
        pie_add_options = [{'label': option, 'value': option} for option in
                           [c for c in ['None'] + categorical_cols if c.lower() not in columns_to_skip_in_UI]]

        # Clear/hide graph
        empty_fig = go.Figure()
        graph_style = {'display': 'none', 'height': '90vh'}

        # if not (stored_dataframe):
        #     data_columns = []
        #     data_data = []
        # data_columns = [{'name': col, 'id': col} for col in stored_dataframe[0].keys()]

        return (
            # df.to_json(date_format='iso', orient='split'),/
            stored_dataframe,
            f"Current file: {filename}",
            original_cols,
            row_variable,
            column_variable,
            numeric_cols,
            categorical_cols,
            encoded_categorical_cols,
            parallel_vars_listbox_options,
            parallel_color_options,
            next((opt['value'] for opt in parallel_color_options if opt['value'].lower().startswith('yield')), 'None'),
            bubble_xy_options,  # bubble-x
            next((opt['value'] for opt in bubble_xy_options if opt['value'].lower() == 'column'), bubble_xy_options[0]),
            bubble_xy_options,  # bubble-y
            next((opt['value'] for opt in bubble_xy_options if opt['value'].lower() == 'row'), bubble_xy_options[0]),
            bubble_options,  # Size
            next((opt['value'] for opt in bubble_options if opt['value'].lower().startswith('yield')), 'None'),
            bubble_options,  # Symbol
            bubble_options,  # Color
            next((opt['value'] for opt in bubble_options if opt['value'].lower().startswith('yield')), 'None'),
            heatmap_color_options,
            next((opt['value'] for opt in heatmap_color_options if opt['value'].lower().startswith('yield')),
                 heatmap_color_options[0]),
            heatmap_add_options,
            heatmap_add_options,
            pie_vars_listbox_options,
            pie_add_options,
            pie_add_options,
            data_data,
            data_columns,
            {'display': 'block'},
            style_cell_conditional,
            empty_fig,
            graph_style,
            "File successfully loaded."
        )
    except Exception as e:
        return [dash.no_update] * 32 + (f"❌ Error processing file: {e}",)


# Generate figure on Generate button click
@app.callback(
    Output('graph-output', 'figure', allow_duplicate=True),
    Output('graph-output', 'style', allow_duplicate=True),
    Output('status-bar', 'children', allow_duplicate=True),
    Input('generate-button', 'n_clicks'),
    State('stored-dataframe', 'data'),
    State('graph-type', 'value'),
    State('original-cols-store', 'data'),
    State('row-variable-store', 'data'),
    State('column-variable-store', 'data'),
    State('numeric-cols-store', 'data'),
    State('categorical-cols-store', 'data'),
    State('encoded-categorical-cols-store', 'data'),
    State({'type': 'listbox', 'subtype': 'parallel-vars'}, 'value'),
    State('parallel-color', 'value'),
    State('bubble-x', 'value'),
    State('bubble-y', 'value'),
    State('bubble-size', 'value'),
    State('bubble-symbol', 'value'),
    State('bubble-color', 'value'),
    State('heatmap-color', 'value'),
    State('heatmap-add-row', 'value'),
    State('heatmap-add-col', 'value'),
    State('heatmap-smooth', 'value'),
    State({'type': 'listbox', 'subtype': 'pie-vars'}, 'value'),
    State('pie-add-row', 'value'),
    State('pie-add-col', 'value'),
    State('pie-norm', 'value'),
    State('pie-norm-value', 'value'),
    State('color-scale', 'value'),
    State('graph-title', 'value'),
    prevent_initial_call=True
)
# def render_figure(n_clicks, stored_data,graph_type):
def render_figure(n_clicks, stored_data, graph_type, original_cols, row_variable, column_variable, numeric_cols,
                  categorical_cols, encoded_categorical_cols, parallel_vars, parallel_color, bubble_x, bubble_y,
                  bubble_size, bubble_symbol, bubble_color, heatmap_color, heatmap_add_row, heatmap_add_col,
                  heatmap_smooth, pie_vars, pie_add_row, pie_add_col, pie_norm, pie_norm_value, color_scale,
                  graph_title):
    if n_clicks is None or n_clicks == 0 or stored_data is None:
        # Don't generate anything on initial load
        return go.Figure(), {'display': 'none'}, "No options selected."

    df = pd.read_json(io.StringIO(stored_data), orient='split')
    original_cols, numeric_cols, categorical_cols, encoded_categorical_cols, row_variable, column_variable = update_dataframe_data(
        df, category_suffix)
    status_to_return = ""

    style = {'display': 'block', 'height': '90vh'}  # base style
    if (graph_type == 'Parallel Coordinates'):
        if not parallel_vars or not parallel_color or parallel_color == 'None':
            return go.Figure(), {'display': 'none'}, "Need to select parallel variables and color variable."
        style['width'] = '90%'
        return generate_parallel_coordinates_graph(df, parallel_vars, parallel_color, color_scale, graph_title,
                                                   category_suffix), style, "Generated parallel coordinates graph."
    elif (graph_type == 'Scatter'):
        if not bubble_x or not bubble_y:
            return go.Figure(), {'display': 'none'}, "Need to select x and y variables."
        style['width'] = '90%'
        return generate_scatter_bubble_graph(df, bubble_x, bubble_y, bubble_size, bubble_symbol, bubble_color,
                                             color_scale, graph_title,
                                             category_suffix), style, "Generated scatter/bubble graph."
    elif (graph_type == 'Heatmap'):
        if not heatmap_color:
            return go.Figure(), {'display': 'none'}, "Need to select color variable."
        style['width'] = '80%'
        if check_row_column_provided(row_variable, column_variable, 'Heatmap'):
            additional_row_variable, additional_column_variable = check_additional_row_column_variables(df,
                                                                                                        row_variable,
                                                                                                        column_variable,
                                                                                                        heatmap_add_row,
                                                                                                        heatmap_add_col)
            if (additional_row_variable != heatmap_add_row) and (additional_column_variable != heatmap_add_col):
                status_to_return = "Specified additional row and column variables are inconsistent in the dataframe and have been ignored;"
            elif (additional_row_variable != heatmap_add_row):
                status_to_return = "Specified additional row variable is inconsistent in the dataframe and have been ignored;"
            elif (additional_column_variable != heatmap_add_col):
                status_to_return = "Specified additional column variable is inconsistent in the dataframe and have been ignored;"
            if not status_to_return:
                status_to_return = "Generated heatmap graph."
            else:
                status_to_return = status_to_return + " generated heatmap graph."
            return generate_heatmap_graph(df, row_variable, column_variable, heatmap_color, additional_row_variable,
                                          additional_column_variable, heatmap_smooth, color_scale,
                                          graph_title), style, status_to_return
        else:
            return go.Figure(), {
                'display': 'none'}, "Heatmap graphs require a valid 'Row' and 'Column' columns in the dataframe/provided data file."
    elif (graph_type == 'Pie Charts'):
        if not pie_vars:
            return go.Figure(), {'display': 'none'}, "Need to select pie chart variables."
        style['width'] = '80%'
        if check_row_column_provided(row_variable, column_variable, 'Pie charts'):
            additional_row_variable, additional_column_variable = check_additional_row_column_variables(df,
                                                                                                        row_variable,
                                                                                                        column_variable,
                                                                                                        pie_add_row,
                                                                                                        pie_add_col)
            if (additional_row_variable != pie_add_row) and (additional_column_variable != pie_add_col):
                status_to_return = "Specified additional row and column variables are inconsistent in the dataframe and have been ignored;"
            elif (additional_row_variable != pie_add_row):
                status_to_return = "Specified additional row variable is inconsistent in the dataframe and have been ignored;"
            elif (additional_column_variable != pie_add_col):
                status_to_return = "Specified additional row variable is inconsistent in the dataframe and have been ignored;"
            if not status_to_return:
                status_to_return = "Generated pie charts graph."
            else:
                status_to_return = status_to_return + " generated pie charts graph."
            if (pie_norm == "Normalize to 100%"):
                pie_norm_index = 0
            elif (pie_norm == 'Values are %'):
                pie_norm_index = 1
            elif (pie_norm == 'Normaliza to value'):
                pie_norm_index = 2
                if not pie_norm_value:
                    return go.Figure(), {'display': 'none'}, "Normalize to value requires a value for pie charts."
            return generate_piecharts_graph(df, row_variable, column_variable, pie_vars, additional_row_variable,
                                            additional_column_variable, pie_norm_index, pie_norm_value, color_scale,
                                            graph_title), style, status_to_return
        else:
            return go.Figure(), {
                'display': 'none'}, "Pie charts graphs require a valid 'Row' and 'Column' columns in the dataframe/provided data file."
    else:
        return go.Figure(), {'display': 'none'}


# Handle download
@app.callback(
    Output("download-graph-html", "data"),
    Input("download-button", "n_clicks"),
    State("graph-output", "figure"),
    prevent_initial_call=True,
)
def download_graph_to_html(n_clicks, figure):
    if not figure:
        return dash.no_update

    fig = Figure(figure)
    import plotly.io as pio
    # fig = pio.from_json(figure)
    html_str = pio.to_html(fig, full_html=True)
    return dict(content=html_str, filename="plotly_graph.html")


# Run the app
if __name__ == '__main__':
    app.run(port=8050, debug=True)
