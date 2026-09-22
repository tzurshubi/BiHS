"""
GuiViewer: standalone player for GuiLogger search traces.

Reads a JSON trace file produced by GuiLogger and replays the search
expansion-by-expansion: nodes are placed on the domain layout (base graph
edges are not drawn), the currently expanded path is highlighted in thick
red, and the successors generated from that expansion branch off the
frontier node as dashed blue lines, each annotated with its f value. Use
Previous/Next to step manually, or Fast Forward to auto-advance one step
per second.

Usage:
    python gui_viewer.py [trace_file.json]

If no trace file is given (e.g. when run directly from an IDE), a file
picker is shown, pre-filtered to GuiLogger trace files, so the user can
choose which one to replay.
"""
import argparse
import glob
import json
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx

# src/utils/gui_viewer.py -> repo root is two levels up
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Must match GuiLogger.DEFAULT_TRACE_DIR in gui_logger.py
TRACE_DIR = os.path.join(REPO_ROOT, "gui_traces")


def find_trace_files(search_dir=TRACE_DIR):
    """All GuiLogger trace files under search_dir, most recent first."""
    pattern = os.path.join(search_dir, "**", "gui_trace_*.json")
    return sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)


def choose_trace_file(search_dir=TRACE_DIR):
    """
    Show a picker listing the relevant trace files under search_dir and
    return the one the user selects (or None if they cancel).
    """
    os.makedirs(search_dir, exist_ok=True)

    try:
        import tkinter as tk
    except ImportError:
        return _choose_trace_file_cli(search_dir)

    candidates = find_trace_files(search_dir)
    selected = {"path": None}

    root = tk.Tk()
    root.title("Select a GuiLogger trace file")
    root.geometry("900x600")

    tk.Label(
        root, text="GuiLogger trace files:", font=("TkDefaultFont", 12, "bold"),
    ).pack(anchor="w", padx=10, pady=(10, 0))

    list_frame = tk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=10, pady=10)

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        list_frame, yscrollcommand=scrollbar.set,
        font=("TkDefaultFont", 11), selectmode="browse",
    )
    for path in candidates:
        listbox.insert("end", os.path.relpath(path, search_dir))
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    if candidates:
        listbox.selection_set(0)

    def on_open(event=None):
        selection = listbox.curselection()
        if selection:
            selected["path"] = candidates[selection[0]]
        root.destroy()

    def on_cancel():
        root.destroy()

    listbox.bind("<Double-Button-1>", on_open)

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))
    tk.Button(btn_frame, text="Cancel", width=12, command=on_cancel).pack(side="right", padx=(5, 0))
    tk.Button(btn_frame, text="Open", width=12, command=on_open).pack(side="right")

    root.mainloop()
    return selected["path"]


def _choose_trace_file_cli(search_dir):
    """Fallback picker for environments without tkinter."""
    candidates = find_trace_files(search_dir)
    if not candidates:
        print(f"No gui_trace_*.json files found under {search_dir}")
        return None

    print("Available trace files:")
    for i, path in enumerate(candidates):
        print(f"  [{i}] {os.path.relpath(path, search_dir)}")

    choice = input("Select a file by number (blank to cancel): ").strip()
    if not choice:
        return None
    try:
        return candidates[int(choice)]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None


def load_trace(path):
    with open(path, "r") as fh:
        return json.load(fh)


def build_graph(trace):
    G = nx.Graph()
    G.add_nodes_from(trace["graph"]["nodes"])
    G.add_edges_from(trace["graph"]["edges"])
    return G


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def grid_positions(rows, cols):
    """
    Node `i * cols + j` sits at row i, column j, matching the indexing
    used by save_table_as_png() in main.py. Row 0 is drawn at the top.
    """
    return {i * cols + j: (j, -i) for i in range(rows) for j in range(cols)}


class GuiViewer:
    def __init__(self, trace_path, layout="spring"):
        self.trace = load_trace(trace_path)
        self.graph = build_graph(self.trace)
        self.steps = self.trace.get("steps", [])
        self.step_index = 0

        meta = self.trace.get("meta", {})
        self.rows = meta.get("rows")
        self.cols = meta.get("cols")
        self.blocked_cells = meta.get("blocked_cells") or []
        self.is_grid = meta.get("graph_type") in ("grid", "maze") and self.rows and self.cols

        if self.is_grid:
            self.pos = grid_positions(self.rows, self.cols)
            figsize = (max(6, self.cols), max(6, self.rows))
        elif layout == "kamada_kawai":
            self.pos = nx.kamada_kawai_layout(self.graph)
            figsize = (9, 8)
        else:
            self.pos = nx.spring_layout(self.graph, seed=42)
            figsize = (9, 8)

        self.fig, self.ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.15)

        ax_back = plt.axes([0.08, 0.02, 0.14, 0.06])
        ax_prev = plt.axes([0.28, 0.02, 0.14, 0.06])
        ax_next = plt.axes([0.44, 0.02, 0.14, 0.06])
        ax_play = plt.axes([0.60, 0.02, 0.18, 0.06])
        self.btn_back = Button(ax_back, "Back")
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_play = Button(ax_play, "Fast Forward")
        self.btn_back.on_clicked(self.on_back)
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        self.btn_play.on_clicked(self.on_toggle_play)

        self._playing = False
        self._play_timer = self.fig.canvas.new_timer(interval=1000)
        self._play_timer.add_callback(self._on_timer_tick)
        self._go_back = False

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw_step()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def on_back(self, event=None):
        self._stop_playing()
        self._go_back = True
        plt.close(self.fig)

    def on_prev(self, event=None):
        self._stop_playing()
        if self.step_index > 0:
            self.step_index -= 1
            self.draw_step()

    def on_next(self, event=None):
        self._stop_playing()
        if self.step_index < len(self.steps) - 1:
            self.step_index += 1
            self.draw_step()

    def on_key(self, event):
        if event.key == "right":
            self.on_next()
        elif event.key == "left":
            self.on_prev()

    # ------------------------------------------------------------------
    # Fast forward (auto-play)
    # ------------------------------------------------------------------
    def _on_timer_tick(self):
        if self.step_index >= len(self.steps) - 1:
            self._stop_playing()
            return
        self.step_index += 1
        self.draw_step()

    def on_toggle_play(self, event=None):
        if self._playing:
            self._stop_playing()
        else:
            self._start_playing()

    def _start_playing(self):
        if self.step_index >= len(self.steps) - 1:
            return
        self._playing = True
        self.btn_play.label.set_text("Pause")
        self._play_timer.start()

    def _stop_playing(self):
        if not self._playing:
            return
        self._playing = False
        self.btn_play.label.set_text("Fast Forward")
        self._play_timer.stop()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_grid_backdrop(self, ax):
        """
        The grid-cell backdrop from save_table_as_png() in main.py: a
        bordered square per cell, filled black for blocked cells.
        """
        blocked = set(self.blocked_cells)
        for i in range(self.rows):
            for j in range(self.cols):
                idx = i * self.cols + j
                face = "black" if idx in blocked else "white"
                rect = patches.Rectangle(
                    (j - 0.5, -i - 0.5), 1, 1,
                    linewidth=1, edgecolor="lightgray", facecolor=face, zorder=0,
                )
                ax.add_patch(rect)
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-self.rows + 0.5, 0.5)

    def draw_step(self):
        ax = self.ax
        ax.clear()

        if self.is_grid:
            self._draw_grid_backdrop(ax)

        step = self.steps[self.step_index] if self.steps else None
        path = step["state"]["path"] if step else []
        head = path[-1] if path else None

        path_edges = list(zip(path, path[1:]))

        node_colors = ["red" if node == head else "skyblue" for node in self.graph.nodes()]

        # Grid domains already show adjacency via the cell backdrop, so the
        # base graph edges are just clutter there and are left undrawn. Other
        # domains (e.g. cubes) have no such backdrop, so their base edges are
        # drawn in neutral gray for structural context.
        if not self.is_grid:
            nx.draw_networkx_edges(
                self.graph, self.pos, ax=ax, edge_color="lightgray", width=1,
            )

        nx.draw_networkx_nodes(
            self.graph, self.pos, ax=ax, node_size=500, node_color=node_colors,
        )
        nx.draw_networkx_labels(
            self.graph, self.pos, ax=ax, font_size=10, font_weight="bold",
        )
        nx.draw_networkx_edges(
            self.graph, self.pos, ax=ax, edgelist=path_edges,
            edge_color="red", width=4,
        )

        if self.is_grid:
            # nx.draw() re-autoscales the axes; pin them back to the grid.
            ax.set_aspect("equal")
            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(-self.rows + 0.5, 0.5)

        if not step:
            ax.set_title("No steps recorded in this trace.")
            self.fig.canvas.draw_idle()
            return

        state = step["state"]

        # f annotation next to the frontier (current head) node
        hx, hy = self.pos[head]
        ax.annotate(
            f"f={state['f']}", (hx, hy),
            textcoords="offset points", xytext=(10, 10),
            fontsize=9, color="darkred", weight="bold",
        )

        # Successors: dashed blue lines branching off the frontier node
        for succ in step["successors"]:
            succ_path = succ["path"]
            branch_at = common_prefix_len(path, succ_path)
            branch_vertices = succ_path[max(branch_at - 1, 0):]
            if len(branch_vertices) < 2:
                continue

            branch_edges = list(zip(branch_vertices, branch_vertices[1:]))
            nx.draw_networkx_edges(
                self.graph, self.pos, ax=ax, edgelist=branch_edges,
                edge_color="blue", width=2, style="dashed",
            )
            new_vertices = branch_vertices[1:]
            nx.draw_networkx_nodes(
                self.graph, self.pos, ax=ax, nodelist=new_vertices,
                node_color="orange", node_size=500,
            )

            new_head = succ_path[-1]
            nhx, nhy = self.pos[new_head]
            ax.annotate(
                f"f={succ['f']}", (nhx, nhy),
                textcoords="offset points", xytext=(10, -12),
                fontsize=8, color="blue",
            )

        meta = self.trace.get("meta", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
        graph_name = self.trace["graph"].get("name", "")
        ax.set_title(
            f"{graph_name}   |   step {self.step_index + 1}/{len(self.steps)}"
            + (f"\n{meta_str}" if meta_str else ""),
            fontsize=10,
        )
        ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def show(self):
        """
        Blocks until the window is closed. Returns the path of a newly chosen
        trace file if the user clicked "Back" (and picked one), else None.
        """
        plt.show()
        if self._go_back:
            return choose_trace_file()
        return None


def main():
    parser = argparse.ArgumentParser(description="Replay a GuiLogger search trace.")
    parser.add_argument(
        "trace_file", nargs="?", default=None,
        help="Path to a JSON trace file produced by GuiLogger. If omitted, a file picker is shown.",
    )
    parser.add_argument(
        "--layout", choices=["spring", "kamada_kawai"], default="spring",
        help="Graph layout algorithm used to position nodes.",
    )
    args = parser.parse_args()

    trace_file = args.trace_file or choose_trace_file()
    if not trace_file:
        print("No trace file selected.")
        return

    while trace_file:
        viewer = GuiViewer(trace_file, layout=args.layout)
        trace_file = viewer.show()


if __name__ == "__main__":
    main()
