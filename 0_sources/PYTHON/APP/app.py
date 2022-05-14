# Run this app with `python3 app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output


def main():

    app = Dash(__name__)

    # assume you have a "long-form" data frame
    # see https://plotly.com/python/px-arguments/ for more options
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "display": "inline-block"
    }

    link_list = [
    dbc.NavLink("Home",             href="/", active="exact"),
    dbc.NavLink("OBR",              href="/obr", active="exact"),
    dbc.NavLink("Measures",         href="/measures", active="exact"),
    dbc.NavLink("Curing process",   href="/curing_process", active="exact"),
    ]


    sidebar = html.Div(
        id="sidebar",
        children=[
            html.H2("Sidebar", className="display-4"),
            html.Hr(),
            html.P("A simple sidebar layout with navigation links", className="lead"),
            dbc.Nav(
                link_list,
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    maindiv = html.Div(
        id="main-div",
        children=[

            # first row
            html.Div([
                html.H2("First Row"),
                html.Hr(),
                html.P(
                    "First row stuff", className="lead"
                )
            ]),

            # second row
            html.Div([
                html.H2("Second Row"),
                html.Hr(),
                html.P(
                    "Second row stuff", className="lead"
                )

            ]),

            # Main plot graph
            dcc.Graph(
                id='example-graph',
                figure=fig
            )
        ],
        style=CONTENT_STYLE
    )



    #content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div([dcc.Location(id="url"), sidebar, maindiv])


    @app.callback(
    Output("sidebar", "style"),
    Output("main-div", "children"),
    Input("url", "pathname"),
    )
    def render_page_content(pathname):
        if pathname == "/":
            return {"backgroundColor": "blue"}
        elif pathname == "/obr":
            return {"backgroundColor": "green"}
        elif pathname == "/measures":
            return {"backgroundColor": "purple"}

        elif pathname == "/curing_process":
            return {"backgroundColor": "purple"}
        # If the user tries to reach a different page, return a 404 message
        return {"backgroundColor": "red"}, dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )


    app.run_server(debug=True)
