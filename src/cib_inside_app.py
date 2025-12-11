import json

from dash import Dash, html, dcc, Input, Output, State, callback_context
from dash.dependencies import ALL

# ---------- Helpers ----------

def hamming_distance(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def check_snake_violation(visited, new_vertex: str, start_vertex: str,
                          ignore_start_neighbor: bool = False) -> bool:
    """
    Snake in the box constraint:
    - No vertex may repeat.
    - No nonconsecutive pair of vertices in the path may have Hamming distance 1.
    If ignore_start_neighbor is True, we ignore chords between new_vertex
    and start_vertex. This is used when new_vertex is a neighbor of 0...0,
    so that such a move does not end the game.
    """
    if not visited:
        return False

    # Revisit of any previous vertex is not allowed
    if new_vertex in visited:
        return True

    last_vertex = visited[-1]

    for v in visited[:-1]:
        if ignore_start_neighbor and v == start_vertex:
            continue
        if hamming_distance(v, new_vertex) == 1:
            # chord between nonconsecutive vertices
            return True

    # Edge between last_vertex and new_vertex has distance 1, and that is allowed
    return False


def is_neighbor_of_zero(vertex: str) -> bool:
    """True if vertex has Hamming weight 1 (neighbor of 0...0)."""
    return vertex.count("1") == 1


def make_dimension_buttons(d, bits, danger_indices=None, neighbor_indices=None):
    """Create the row of d circular buttons, one per dimension."""
    if danger_indices is None:
        danger_indices = set()
    else:
        danger_indices = set(danger_indices)

    if neighbor_indices is None:
        neighbor_indices = set()
    else:
        neighbor_indices = set(neighbor_indices)

    buttons = []
    for i in range(d):
        is_on = bits[i] == 1
        bg_color = "#4a4a4a" if is_on else "#e0e0e0"
        text_color = "#ffffff" if is_on else "#000000"

        if i in danger_indices:
            border_style = "4px solid red"
        elif i in neighbor_indices:
            border_style = "4px solid #ffcc00"  # yellow
        else:
            border_style = "3px solid #333333"

        buttons.append(
            html.Button(
                str(i),
                id={"type": "dim-button", "index": i},
                n_clicks=0,
                style={
                    "display": "inline-block",
                    "margin": "12px",
                    "width": f"{DEFAULT_CIRCLE_RADIUS * 2}px",
                    "height": f"{DEFAULT_CIRCLE_RADIUS * 2}px",
                    "borderRadius": "50%",
                    "lineHeight": "90px",
                    "textAlign": "center",
                    "fontSize": "26px",
                    "cursor": "pointer",
                    "border": border_style,
                    "backgroundColor": bg_color,
                    "color": text_color,
                },
            )
        )
    return buttons


def initial_state(d):
    start_vertex = "0" * d
    return {
        "bits": [0] * d,
        "path_length": 0,
        "visited": [start_vertex],
        "game_over": False,
    }


# ---------- App ----------

app = Dash(__name__)
server = app.server  # for Hugging Face / gunicorn

DEFAULT_D = 4
DEFAULT_CIRCLE_RADIUS = 60

app.layout = html.Div(
    style={
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        "maxWidth": "900px",
        "margin": "0 auto",
        "padding": "30px",
    },
    children=[
        html.H2("Hypercube Coil Game"),

        html.P(
            "Choose the dimension d, then click the circles. "
            "Each click flips the bit in that dimension in the current vertex. "
            "You always start from 0…0. "
            "Red bordered circles are moves that will immediately break the snake constraint. "
            "Yellow bordered circles go to vertices that are neighbors of 0…0, "
            "which you may want for closing a coil later."
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
            data=initial_state(DEFAULT_D),
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
            style={
                "marginTop": "30px",
                "display": "flex",
                "gap": "40px",
                "flexWrap": "wrap",
                "alignItems": "center",
            },
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

        # Game over banner placeholder
        html.Div(
            id="game-over-banner",
            style={"marginTop": "25px"},
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
    - Changing d or pressing "new game" resets bits, path length, visited, and game_over.
    - Clicking a dimension button toggles that bit, advances in the hypercube,
      and checks the snake constraint, with a special allowance for moves that
      go to a neighbor of 0...0.
    """
    ctx = callback_context

    if state is None:
        return initial_state(dimension)

    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Reset on slider change or new game
    if trigger in ("dimension-slider", "new-game-button") or trigger is None:
        return initial_state(dimension)

    # If the game is already over, ignore further clicks
    if state.get("game_over", False):
        return state

    # A dimension button was pressed
    try:
        trigger_id = json.loads(trigger)
        idx = trigger_id["index"]
    except Exception:
        return state

    bits = state.get("bits", [0] * dimension)
    path_length = state.get("path_length", 0)
    visited = state.get("visited", [])
    game_over = state.get("game_over", False)

    # In case dimension changed but state was not reset properly
    if len(bits) != dimension:
        return initial_state(dimension)

    if isinstance(idx, int) and 0 <= idx < dimension:
        # Compute the new vertex
        new_bits = bits.copy()
        new_bits[idx] = 1 - new_bits[idx]
        new_vertex = "".join(str(b) for b in new_bits)

        start_vertex = "0" * dimension
        neighbor_zero = is_neighbor_of_zero(new_vertex)

        violation = check_snake_violation(
            visited,
            new_vertex,
            start_vertex,
            ignore_start_neighbor=neighbor_zero,
        )

        # Apply the move
        bits = new_bits
        path_length += 1
        visited = visited + [new_vertex]

        if violation:
            game_over = True

    return {
        "bits": bits,
        "path_length": path_length,
        "visited": visited,
        "game_over": game_over,
    }


@app.callback(
    Output("buttons-container", "children"),
    Output("path-length-display", "children"),
    Output("current-vertex-display", "children"),
    Output("game-over-banner", "children"),
    Input("dimension-slider", "value"),
    Input("game-state", "data"),
)
def refresh_view(dimension, state):
    """
    Render the buttons, path length, current vertex, danger moves, neighbor of zero moves,
    and game over banner.
    """
    if state is None:
        state = initial_state(dimension)

    bits = state.get("bits", [0] * dimension)
    path_length = state.get("path_length", 0)
    visited = state.get("visited", [])
    game_over = state.get("game_over", False)

    if len(bits) != dimension:
        bits = [0] * dimension

    start_vertex = "0" * dimension

    danger_indices = []
    neighbor_indices = []

    if visited:
        for i in range(dimension):
            test_bits = bits.copy()
            test_bits[i] = 1 - test_bits[i]
            test_vertex = "".join(str(b) for b in test_bits)
            neighbor_zero = is_neighbor_of_zero(test_vertex)

            violation = check_snake_violation(
                visited,
                test_vertex,
                start_vertex,
                ignore_start_neighbor=neighbor_zero,
            )

            if violation:
                danger_indices.append(i)
            elif neighbor_zero:
                neighbor_indices.append(i)

    buttons = make_dimension_buttons(
        dimension,
        bits,
        danger_indices=danger_indices,
        neighbor_indices=neighbor_indices,
    )
    bit_string = "".join(str(b) for b in bits)

    path_text = f"Current path length: {path_length}"
    vertex_text = f"Current vertex: {bit_string}"

    if game_over:
        banner = html.Div(
            "GAME OVER - Snake constraint violated",
            style={
                "backgroundColor": "#ff4d4d",
                "color": "white",
                "padding": "16px",
                "borderRadius": "8px",
                "textAlign": "center",
                "fontSize": "24px",
                "fontWeight": "700",
                "boxShadow": "0 0 12px rgba(255, 0, 0, 0.7)",
            },
        )
    else:
        banner = ""

    return buttons, path_text, vertex_text, banner


if __name__ == "__main__":
    app.run_server(debug=True)
