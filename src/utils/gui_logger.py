import json
import os
from datetime import datetime

# src/utils/gui_logger.py -> repo root is two levels up
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "gui_traces")


def _materialize(path_like):
    """
    Accepts either a State (with .materialize_path()) or a plain list/tuple
    of vertices, and returns a plain list of vertices.
    """
    if hasattr(path_like, "materialize_path"):
        return path_like.materialize_path()
    return list(path_like)


class GuiLogger:
    """
    Trace-Driven Architecture tracer (the "Tracer" half of GuiLogger/GuiViewer).

    Records search expansion steps to a JSON trace file so search dynamics can
    be replayed later with GuiViewer, without slowing down the search itself.

    When video=False every method is a no-op: gui_log_expansion() returns
    immediately, so there is zero overhead on the search loop.
    """

    def __init__(self, video, graph, graph_name, algo_name, output_dir=DEFAULT_TRACE_DIR, meta=None, search_type=None, direction=None):
        self.video = bool(video)
        self.output_path = None
        if not self.video:
            return

        self.graph_name = graph_name
        self.algo_name = algo_name
        self.search_type = search_type
        self.direction = direction
        self._steps = []
        self.output_path = self._build_output_path(output_dir)

        self._trace = {
            "graph": {
                "name": graph_name,
                "nodes": list(graph.nodes),
                "edges": [list(e) for e in graph.edges],
            },
            "meta": dict(meta) if meta else {},
            "steps": self._steps,
        }

    def _build_output_path(self, output_dir):
        safe_graph_name = str(self.graph_name).replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label = f"{self.algo_name}_{self.search_type}" if self.search_type else str(self.algo_name)
        if self.direction:
            label += f"_{self.direction}"
        filename = f"gui_trace_{safe_graph_name}_{label}_{timestamp}.json"
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def gui_log_expansion(self, state, f, h, successors):
        """
        Record one expansion event.

        state: the state being expanded (a State object, or a plain list of
            vertices).
        f, h: the f and h values of `state` at the time of expansion.
        successors: iterable of (successor, f, h) tuples, where `successor`
            is a State object or a plain list of vertices, and f/h are its
            updated values.
        """
        if not self.video:
            return

        self._steps.append({
            "index": len(self._steps),
            "state": {
                "path": _materialize(state),
                "f": f,
                "h": h,
            },
            "successors": [
                {"path": _materialize(succ), "f": succ_f, "h": succ_h}
                for succ, succ_f, succ_h in successors
            ],
        })

    def save(self):
        """Write the accumulated trace to disk. No-op when video=False."""
        if not self.video:
            return None

        with open(self.output_path, "w") as fh:
            json.dump(self._trace, fh)
        return self.output_path
