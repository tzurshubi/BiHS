import json

from dash import Dash, html, dcc, Input, Output, State, callback_context
from dash.dependencies import ALL

# ---------- Helpers ----------

def make_dimension_buttons(d, bits):
    """Create the row of d circular buttons, one per dimension."""
    buttons = []
    for i in range(d):
        is_on = bits[i] == 1
        bg_color = "#4a4a4a" if is_on else "#e0e0e0"
        text_color = "#ffffff" if is_on else "#000000"

        buttons.append(
            html.Button(
                str(i),
                id={"type": "dim-button", "index": i},
                n_clicks=0,
                style={
                    "display": "inline-block",
                    "margin": "10px",
                    "width": "60px",
                    "height": "60px",
                    "borderRadius": "50%",
                    "lineHeight": "60px",
                    "textAlign": "center",
                    "fontSize": "22px",
                    "cursor": "pointer",
                    "border": "2px solid #333333",
                    "backgroundColor": bg_color,
                    "color": text_color,
                },
            )
        )
    return buttons


# ---------- App ----------

app = Dash(__name__)
server = app.server  # for Hugging Face / gunicorn

DEFAULT_D = 3

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
           "maxWidth": "900px",
           "margin": "0 auto",
           "padding": "30px"},
    children=[
        html.H2("Hypercube Coil Game"),

        html.P(
            "Choose the dimension d, then click the circles. "
            "Each click flips the bit in that dimension in the current vertex. "
            "You always start from 0…0."
        ),

        # Dimension slider
        html.Div(
            style={"marginTop": "20px", "marginBottom": "20px"},
            children=[
                html.Label("Dimension d"),
                dcc.Slider(
                    id="dimension-slider",
                    min=1,
                    max=12,
                    step=1,
                    value=DEFAULT_D,
                    marks={i: str(i) for i in range(1, 13)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
        ),

        # Game state store
        dcc.Store(
            id="game-state",
            data={"bits": [0] * DEFAULT_D, "path_length": 0},
        ),

        # Main buttons row
        html.Div(
            id="buttons-container",
            style={
                "display": "flex",
                "justifyContent": "center",
                "flexWrap": "wrap",
                "marginTop": "40px",
            },
            children=make_dimension_buttons(DEFAULT_D, [0] * DEFAULT_D),
        ),

        # Info row
        html.Div(
            style={"marginTop": "30px", "display": "flex", "gap": "40px",
                   "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div(
                    id="path-length-display",
                    style={"fontSize": "18px", "fontWeight": "600"},
                ),
                html.Div(
                    id="current-vertex-display",
                    style={"fontSize": "18px"},
                ),
                html.Button(
                    "Start new game",
                    id="new-game-button",
                    n_clicks=0,
                    style={
                        "padding": "8px 16px",
                        "fontSize": "16px",
                        "cursor": "pointer",
                        "borderRadius": "6px",
                    },
                ),
            ],
        ),
    ],
)


# ---------- Callbacks ----------

@app.callback(
    Output("game-state", "data"),
    Input("dimension-slider", "value"),
    Input("new-game-button", "n_clicks"),
    Input({"type": "dim-button", "index": ALL}, "n_clicks"),
    State("game-state", "data"),
)
def update_game_state(dimension, new_game_clicks, button_clicks, state):
    """
    Central state update:
    - Changing d or pressing "new game" resets bits and path length.
    - Clicking a dimension button toggles that bit and increases path length by 1.
    """
    ctx = callback_context

    # On very first load
    if state is None:
        return {"bits": [0] * dimension, "path_length": 0}

    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Reset on slider change or new game
    if trigger in ("dimension-slider", "new-game-button") or trigger is None:
        return {"bits": [0] * dimension, "path_length": 0}

    # Otherwise a dimension button was pressed
    try:
        trigger_id = json.loads(trigger)
        idx = trigger_id["index"]
    except Exception:
        return state

    bits = state.get("bits", [0] * dimension)
    path_length = state.get("path_length", 0)

    # If d changed recently, ensure bits length matches d
    if len(bits) != dimension:
        bits = [0] * dimension
        path_length = 0

    if isinstance(idx, int) and 0 <= idx < dimension:
        bits[idx] = 1 - bits[idx]
        path_length += 1

    return {"bits": bits, "path_length": path_length}


@app.callback(
    Output("buttons-container", "children"),
    Output("path-length-display", "children"),
    Output("current-vertex-display", "children"),
    Input("dimension-slider", "value"),
    Input("game-state", "data"),
)
def refresh_view(dimension, state):
    """Render the buttons, path length, and current vertex from the stored state."""
    if state is None:
        bits = [0] * dimension
        path_length = 0
    else:
        bits = state.get("bits", [0] * dimension)
        path_length = state.get("path_length", 0)

    # Guard if bits length does not match d
    if len(bits) != dimension:
        bits = [0] * dimension

    buttons = make_dimension_buttons(dimension, bits)
    bit_string = "".join(str(b) for b in bits)

    path_text = f"Current path length: {path_length}"
    vertex_text = f"Current vertex: {bit_string}"

    return buttons, path_text, vertex_text


if __name__ == "__main__":
    app.run_server(debug=True)
