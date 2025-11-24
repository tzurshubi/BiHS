import math
from typing import List

from dash import Dash, html, dcc, Input, Output, State, ctx, no_update
import plotly.graph_objects as go


# ------------------------------------------------------------
# Default settings for the Hypercube visualizer
# ------------------------------------------------------------
DEFAULTS = {
    "dimension": 3,          # start with Q4
    "scale": 1600.0,          # base layout scale
    "node_radius": 3,        # node size (radius)
    "edge_width": 2,         # edge thickness
    "show_labels": False,    # show binary labels on nodes # True / False
    "label_fontsize": 12,    # font size of labels
    "max_dimension": 12,     # slider max
    "min_dimension": 1,      # slider min
}



# ---------- Hypercube utilities ----------

def swap_dims_vertex(v: int, i: int, j: int) -> int:
    """
    Swap bits i and j in the vertex id v.
    (Coordinate permutation on Q_d.)
    """
    if i == j:
        return v
    bi = (v >> i) & 1
    bj = (v >> j) & 1
    if bi != bj:
        # flip both bits: 01 -> 10 or 10 -> 01
        v ^= (1 << i) | (1 << j)
    return v


def flip_dim_vertex(v: int, k: int) -> int:
    """
    Flip bit k in the vertex id v.
    (Translation by the unit vector in coordinate k.)
    """
    return v ^ (1 << k)


def swap_dims_path(path, d: int, i: int, j: int):
    """Apply swap_dims_vertex to every vertex in the path."""
    if i < 0 or j < 0 or i >= d or j >= d:
        return path
    return [swap_dims_vertex(v, i, j) for v in path]


def flip_dim_path(path, d: int, k: int):
    """Apply flip_dim_vertex to every vertex in the path."""
    if k < 0 or k >= d:
        return path
    return [flip_dim_vertex(v, k) for v in path]


def classify_path(path, d: int):
    """
    Classify a path in Q_d as one of:
      - "snake"        (induced simple path)
      - "coil"         (induced simple cycle)
      - "almost coil"  (open path that would be a coil if closed)
      - "not snake"    otherwise
    Returns: (label, is_valid)
    """

    if not path or len(path) <= 1:
        return "snake", True  # trivial path treated as snake

    # Is the path explicitly closed (first == last)?
    is_closed = (path[0] == path[-1])

    # Work with a version without duplicate endpoint when closed
    cycle = path[:-1] if is_closed else path[:]
    n = len(cycle)

    # --- 1) all vertices must be distinct ---
    if len(set(cycle)) != n:
        return "not snake", False

    # --- 2) consecutive vertices must be adjacent ---
    for i in range(n - 1):
        if hamming_dist(cycle[i], cycle[i + 1]) != 1:
            return "not snake", False

    # Whether the endpoints (0, n-1) are adjacent in Q_d
    closing_adjacent = (hamming_dist(cycle[0], cycle[-1]) == 1)

    # If the user claims a cycle (is_closed), that closing edge must exist
    if is_closed and not closing_adjacent:
        return "not snake", False

    # --- 3) inducedness: no chords among internal pairs ---
    # We forbid adjacency for any non-consecutive pair (i,j),
    # BUT we always skip (0, n-1) because that edge is either:
    #    - the actual closing edge of a coil, or
    #    - the would-be closing edge of an "almost coil".
    for i in range(n):
        for j in range(i + 1, n):
            # Skip the path edges
            if j == i + 1:
                continue
            # Skip endpoints pair (0, n-1) in both open/closed cases
            if i == 0 and j == n - 1:
                continue
            if hamming_dist(cycle[i], cycle[j]) == 1:
                return "not snake", False

    # --- 4) classification based on closure + endpoint adjacency ---

    if is_closed:
        # distinct, consecutive adjacent, closed by an edge, no chords
        return "coil", True

    # Open path, induced
    if closing_adjacent:
        # Endpoints adjacent → would be a coil if closed
        return "almost coil", True

    return "snake", True


def hamming_dist(a: int, b: int) -> int:
    x, c = a ^ b, 0
    while x:
        c += x & 1
        x >>= 1
    return c


def build_hypercube(d: int):
    n = 1 << d
    nodes = list(range(n))
    edges = []
    for u in range(n):
        for bit in range(d):
            v = u ^ (1 << bit)
            if u < v:
                edges.append((u, v, bit))
    return nodes, edges


def dim_color(k: int) -> str:
    hues = [210, 20, 140, 80, 0, 260, 40, 180, 320, 120]
    h = hues[k % len(hues)]
    return f"hsl({h},65%,45%)"


def int_to_bin(n: int, d: int) -> str:
    return format(n, f"0{d}b")

# ---------- Deterministic 2D layout (no rotation) ----------
# Even bits → X offsets, odd bits → Y offsets, decreasing magnitudes.
def layout_positions(d: int, base: float = 900.0):
    n = 1 << d # number of nodes = 2^d
    dx, dy = [0.0] * d, [0.0] * d # per-dimension offsets
    for k in range(d): # dimension k
        tier = k // 2 # tier 0,1,2,...
        mag = (2 ** max(0, (d - 1) - tier)) * (base / (2 ** d)) # magnitude
        if k % 2 == 0:
            dx[k] = mag
        else:
            dy[k] = mag

    pts = []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for vid in range(n):
        x = sum(dx[k] for k in range(d) if (vid >> k) & 1)
        y = sum(dy[k] for k in range(d) if (vid >> k) & 1)
        if (vid >> 2) & 1:
            x += 100
            y += 250
        if (vid >> 3) & 1:
            x += 1800
            y -= 200
        if (vid >> 4) & 1:
            x += 200
            y += 1800
        if (vid >> 5) & 1:
            x += 3800
            y += 3200
        if (vid >> 6) & 1:
            x += 3400
            y -= 200

        pts.append((vid, x, y))
        minx, maxx = min(minx, x), max(maxx, x)
        miny, maxy = min(miny, y), max(maxy, y)

    pts2 = [(vid, x - minx, y - miny) for vid, x, y in pts]
    width = maxx - minx
    height = maxy - miny
    return pts2, width, height


def longest_cib(d: int):
    """
    Return a predefined coil/snake for certain dimensions,
    otherwise fall back to a standard Gray-code path.
    """
    presets = {
        3: [0, 1, 3, 7, 6, 4, 0],
        4: [0, 1, 3, 7, 6, 14, 10, 8, 0],
        5: [0, 1, 3, 7, 6, 14, 12, 13, 29, 31, 27, 26, 18, 16, 0],
        6: [0, 1, 3, 7, 15, 31, 29, 25, 24, 26, 10, 42, 43, 59,
            51, 49, 53, 37, 45, 44, 60, 62, 54, 22, 20, 4, 0],
    }

    if d in presets:
        return presets[d][:]  # copy, so we don't mutate the original

    # Fallback: Gray-code Hamiltonian path for other dimensions
    n = 1 << d
    seq = []
    for i in range(n):
        g = i ^ (i >> 1)
        seq.append(g)
    return seq



def parse_path(text: str, d: int) -> List[int]:
    if not text:
        return []
    out = []
    for tok in [t for t in text.replace(",", " ").split() if t]:
        if False and all(c in "01" for c in tok): # Disabled bit representation
            val = int(tok, 2)
        elif tok.isdigit():
            val = int(tok)
        else:
            continue
        if 0 <= val < (1 << d):
            out.append(val)
    return out

# ---------- Figure builder (fixed UnboundLocalError) ----------

def make_figure(d: int, show_labels: bool, node_r: int, edge_w: int,
                path: List[int], scale_base: float):
    nodes, edges = build_hypercube(d)
    pts, width, height = layout_positions(d, base=scale_base)
    pos = {vid: (x, y) for vid, x, y in pts}

    # path edge set for emphasis
    path_edges = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        key = (a, b) if a < b else (b, a)
        path_edges.add(key)

    # ---- define BEFORE appending to it ----
    edge_traces = []

    # Base edges (clickable), grouped by dimension, customdata carries (u,v)
    for bit in range(d):
        xs, ys, cd = [], [], []
        for (u, v, b) in edges:
            if b != bit:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            xs += [x1, x2, None]
            ys += [y1, y2, None]
            cd += [[u, v], [u, v], None]
        edge_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=edge_w, color=dim_color(bit)),
                opacity=0.35,
                hoverinfo="skip",
                name=f"bit {bit}",
                customdata=cd,   # lets us detect edge clicks
            )
        )

    # Emphasize path edges on top (3×, same colors as original)
    if path_edges:
        xs, ys, colors = [], [], []
        for (u, v, b) in edges:
            key = (u, v) if u < v else (v, u)
            if key in path_edges:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                xs += [x1, x2, None]
                ys += [y1, y2, None]
                colors.append(dim_color(b))  # track color per segment

        # Plot each colored segment individually so they keep their dimension color
        for i, (u, v, b) in enumerate(edges):
            key = (u, v) if u < v else (v, u)
            if key in path_edges:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                edge_traces.append(
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="lines",
                        line=dict(width=max(1, edge_w * 3), color=dim_color(b)),
                        opacity=1.0,
                        hoverinfo="skip",
                        name=f"path bit {b}",
                    )
                )


    # Nodes (3× size if on the path)
    xs = [pos[v][0] for v in nodes]
    ys = [pos[v][1] for v in nodes]
    texts = [int_to_bin(v, d) for v in nodes]

    in_path = set(path or [])
    sizes = []
    colors = []
    for v in nodes:
        base_size = node_r * 2
        sizes.append(base_size * (3 if v in in_path else 1))  # 3× for nodes on the path
        if path and (v == path[0] or v == path[-1]):
            colors.append("#111")
        else:
            colors.append("#222")

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#333")),
        text=texts if show_labels else None,
        textposition="middle right",
        textfont=dict(size=12, color="#333"),
        hovertemplate=(
            "id=%{customdata}<br>bin=%{text}<extra></extra>" if show_labels
            else "id=%{customdata}<extra></extra>"
        ),
        customdata=nodes,  # integers → vertex id for vertex clicks
        name="vertices",
    )

    fig = go.Figure(edge_traces + [node_trace])
    pad = max(40, 0.08 * max(width, height))
    fig.update_layout(
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False, range=[-pad, width + pad]),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, range=[-pad, height + pad]),
        dragmode=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# ---------- Dash App ----------

app = Dash(__name__)
app.title = "Hypercube Path Explorer (Python/Dash)"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("Hypercube Path Explorer (Python/Dash)"),
        html.Div(id="stats", style={"opacity": 0.7, "marginBottom": "10px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px", "marginBottom": "12px"},
            children=[
                html.Div([
                    html.Label("Dimension d (1–12)"),
                    dcc.Slider(
                        id="dim",
                        min=1,
                        max=12,
                        step=1,
                        value=DEFAULTS["dimension"],
                        marks=None,
                        tooltip={"always_visible": True},
                    ),
                ]),
                html.Div([
                    html.Label("Layout scale"),
                    dcc.Slider(
                        id="scale",
                        min=800,          # larger range
                        max=2600,
                        step=50,
                        value=int(DEFAULTS["scale"]),   # 1600 by default
                        marks=None,
                        tooltip={"always_visible": True},
                    ),
                ]),
                html.Div([
                    dcc.Checklist(
                        id="show_labels",
                        options=[{"label": " Show labels", "value": "labels"}],
                        value=["labels"] if DEFAULTS["show_labels"] else [],
                        style={"marginTop": "8px"},
                    )
                ]),
                html.Div(),  # empty cell just to keep the grid tidy
            ],
        ),


        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center", "marginBottom": "8px"},
            children=[
                dcc.Input(id="manual_path",
                          placeholder="Enter path (e.g. 0,1,3)",
                          style={"flex": 1}, debounce=True),
                html.Button("Set path", id="btn_set", n_clicks=0),
                html.Button("Longest CIB", id="btn_longest_cib", n_clicks=0,
                            style={"background": "#059669", "color": "white"}),
                html.Button("Clear", id="btn_clear", n_clicks=0),
            ]
        ),

        # Automorphism controls (swap / flip dimensions)
        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center", "marginBottom": "8px"},
            children=[
                html.Span("Swap dimensions:", style={"fontSize": "0.9rem"}),
                dcc.Input(
                    id="swap_i",
                    type="number",
                    min=0,
                    step=1,
                    value=0,
                    placeholder="i",
                    style={"width": "60px"},
                ),
                dcc.Input(
                    id="swap_j",
                    type="number",
                    min=0,
                    step=1,
                    value=0,
                    placeholder="j",
                    style={"width": "60px"},
                ),
                html.Button("Swap", id="btn_swap", n_clicks=0),

                html.Span("Flip dimension:", style={"fontSize": "0.9rem", "marginLeft": "16px"}),
                dcc.Input(
                    id="flip_k",
                    type="number",
                    min=0,
                    step=1,
                    value=0,
                    placeholder="k",
                    style={"width": "60px"},
                ),
                html.Button("Flip", id="btn_flip", n_clicks=0),
            ],
        ),


        html.Div(
            id="path_info",
            style={
                "marginBottom": "8px",
                "fontFamily": "monospace",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
        ),


        dcc.Graph(id="fig", style={"height": "800px"}, config={"displayModeBar": True}),
        dcc.Store(id="path_store", data=[]),
    ]
)

@app.callback(
    Output("stats", "children"),
    Input("dim", "value"),
    Input("path_store", "data"),
)
def stats(d, path):
    n = 1 << d
    m = d * (1 << (d - 1))
    plen = len(path) if path else 0
    return f"Q_{d} · vertices: {n} · edges: {m} · path length: {max(0, plen - 1)}"

@app.callback(
    Output("path_store", "data"),
    Input("fig", "clickData"),
    Input("btn_clear", "n_clicks"),
    Input("btn_longest_cib", "n_clicks"),
    Input("btn_set", "n_clicks"),
    Input("btn_swap", "n_clicks"),
    Input("btn_flip", "n_clicks"),
    State("path_store", "data"),
    State("manual_path", "value"),
    State("dim", "value"),
    State("swap_i", "value"),
    State("swap_j", "value"),
    State("flip_k", "value"),
    prevent_initial_call=True
)
def update_path(clickData,
                n_clear,
                n_gray,
                n_set,
                n_swap,
                n_flip,
                path,
                manual_text,
                d,
                swap_i,
                swap_j,
                flip_k):
    trigger = ctx.triggered_id
    path = path or []
    d = int(d)


    # 1) Clear
    if trigger == "btn_clear":
        return []

    # 2) Longest CIB
    if trigger == "btn_longest_cib":
        return longest_cib(d)

    # 3) Manual set path
    if trigger == "btn_set":
        newp = parse_path(manual_text or "", d)
        return newp if newp else path

    # 4) Swap two dimensions (apply to entire path)
    if trigger == "btn_swap":
        try:
            i = int(swap_i) if swap_i is not None else None
            j = int(swap_j) if swap_j is not None else None
        except (TypeError, ValueError):
            return path
        if i is None or j is None:
            return path
        # clamp to valid range
        if not (0 <= i < d and 0 <= j < d):
            return path
        return swap_dims_path(path, d, i, j)

    # 5) Flip one dimension (apply to entire path)
    if trigger == "btn_flip":
        try:
            k = int(flip_k) if flip_k is not None else None
        except (TypeError, ValueError):
            return path
        if k is None or not (0 <= k < d):
            return path
        return flip_dim_path(path, d, k)


    # 6) Figure clicks: vertex (int) or edge ([u,v])
    if trigger == "fig" and clickData and clickData.get("points"):
        p = clickData["points"][0]
        cd = p.get("customdata")

        # Vertex click → add if adjacent, else restart
        if isinstance(cd, (int, float)):
            vid = int(cd)

            # If no path yet → start it
            if not path:
                return [vid]

            # If clicked the *last* vertex again → remove it (un-click)
            if vid == path[-1]:
                return path[:-1]

            # If clicked a vertex that is directly before the last one → also allow backtracking
            if len(path) >= 2 and vid == path[-2]:
                return path[:-1]

            # Otherwise, if adjacent to last vertex → extend path
            if hamming_dist(vid, path[-1]) == 1:
                return path + [vid]

            # Otherwise not adjacent → start a new path from this vertex
            return [vid]


        # Edge click → add other endpoint if extending current endpoint
        if isinstance(cd, (list, tuple)) and len(cd) == 2:
            u, v = int(cd[0]), int(cd[1])
            if not path:
                return [u, v]
            last = path[-1]
            if last == u:
                return path + [v]
            if last == v:
                return path + [u]
            return [u, v]

    return path

@app.callback(
    Output("fig", "figure"),
    Input("dim", "value"),
    Input("show_labels", "value"),
    Input("path_store", "data"),
    Input("scale", "value"),
)
def render(d, show_labels_vals, path, scale_base):
    show_labels = "labels" in (show_labels_vals or [])
    fig = make_figure(
        d=int(d),
        show_labels=bool(show_labels),
        node_r=DEFAULTS["node_radius"],
        edge_w=DEFAULTS["edge_width"],
        path=path or [],
        scale_base=float(scale_base),
    )
    return fig

@app.callback(
    Output("path_info", "children"),
    Input("dim", "value"),
    Input("path_store", "data"),
)
def path_info(d, path):
    path = path or []

    if not path:
        return html.Span("Path: (empty)")

    path_str = ", ".join(str(v) for v in path)

    label, valid = classify_path(path, d)

    color = {
        "snake": "green",
        "coil": "green",
        "almost coil": "green",
        "not snake": "red",
    }[label]

    return html.Span([
        html.Span(f"Path: {path_str} "),
        html.Span(f"[{label}]", style={"color": color, "fontWeight": "bold"}),
    ])


if __name__ == "__main__":
    app.run_server(debug=True)
