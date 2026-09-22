"""Regression coverage for starting paths, lookahead, and buffer crossings."""
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from models.state import State
from utils.utils import init_search_stats


NAMES = ('XA', 'XDFBnB', 'XIDA', 'BiXA', 'BiXDFBnB', 'BiXIDA')
MODULES = {name: importlib.import_module('algorithms.' + name) for name in NAMES}
STATE_MODULE = importlib.import_module('models.state')


def make_args(graph, lookahead=2, bsd=False):
    return SimpleNamespace(
        stats=init_search_stats(max(graph.nodes)), logger=lambda *a, **kw: None,
        graph_type='grid', cube_buffer_dim=None, lookahead=lookahead,
        bsd=bsd, sym_coil=False, solution_vertices=[], graph=graph,
        original_graph=graph, gui_logger=SimpleNamespace(video=False),
    )


class SearchStateRegressionTests(unittest.TestCase):
    def test_materialized_descendants_preserve_the_stored_prefix(self):
        for store_path in (False, True):
            for snake in (False, True):
                with self.subTest(store_path=store_path, snake=snake), \
                        patch.object(STATE_MODULE, 'STORE_PATH', store_path):
                    graph = nx.path_graph(6)
                    args = make_args(graph)
                    root = State(graph, [0, 1, 2], snake=snake, args=args)
                    child = root.generate_successors(args, snake)[0]
                    grandchild = child.generate_successors(args, snake)[0]
                    self.assertEqual(child.materialize_path(), [0, 1, 2, 3])
                    self.assertEqual(grandchild.materialize_path(), [0, 1, 2, 3, 4])
                    self.assertEqual(grandchild.tail(), [0, 1, 2, 3])
                    self.assertEqual(grandchild.g, 4)
                    path = grandchild.materialize_path()
                    path[0] = 99
                    self.assertEqual(root.materialize_path(), [0, 1, 2])
                    self.assertEqual(grandchild.materialize_path(), [0, 1, 2, 3, 4])

    def test_searches_keep_initial_paths_in_both_directions(self):
        for name in NAMES:
            for snake in (False, True):
                for bsd in (False, True):
                    for heuristic in (None, 'bcc_heuristic'):
                        with self.subTest(name=name, snake=snake, bsd=bsd, heuristic=heuristic):
                            graph = nx.path_graph(7)
                            goal = [6, 5] if name.startswith('Bi') else 6
                            result = getattr(MODULES[name], name)(
                                graph, [0, 1], goal, heuristic, snake, make_args(graph, bsd=bsd))
                            self.assertEqual(result[0], list(range(7)))

    def test_bixdfbnb_uses_the_requested_lookahead(self):
        module = MODULES['BiXDFBnB']
        for k in (-2, -1, 1, 2, 3, 4, 6):
            with self.subTest(lookahead=k):
                graph = nx.path_graph(17)
                graph.add_edge(0, 17)  # F has more successors than B at the root.
                args = make_args(graph, lookahead=k)
                first_leaf_depths = []
                original_h = module.heuristic

                def h(f, b, *a, **kw):
                    if args.stats['expansions'] == 1:
                        first_leaf_depths.append((f.g, b.g))
                    return original_h(f, b, *a, **kw)

                with patch.object(module, 'heuristic', h):
                    path, _, _ = module.BiXDFBnB(graph, 0, 16, 'bcc_heuristic', False, args)
                self.assertEqual(path, list(range(17)))
                self.assertTrue(first_leaf_depths)
                if k == -2:
                    self.assertEqual(set(first_leaf_depths), {(0, 1)})
                elif k == -1:
                    self.assertEqual(set(first_leaf_depths), {(1, 0)})
                else:
                    self.assertEqual({f + b for f, b in first_leaf_depths}, {k})

    def test_xa_buffer_constraint_matches_exhaustive_search(self):
        graph = nx.Graph((v, v ^ (1 << d)) for v in range(8) for d in range(3))
        scenarios = (([0], 7), ([0], 6), ([0, 1], 7), ([0, 1], 6), ([0, 1, 3, 2], 2))
        for dim in range(3):
            for prefix, goal in scenarios:
                candidates = [
                    p for p in nx.all_simple_paths(graph, prefix[0], goal)
                    if p[:len(prefix)] == prefix
                    and sum((u ^ v) == (1 << dim) for u, v in zip(p, p[1:])) <= 1
                ]
                expected = max(candidates, key=len, default=[])
                for k in (1, 2, 3, 4):
                    for bsd in (False, True):
                        for heuristic in (None, 'bcc_heuristic'):
                            with self.subTest(dim=dim, prefix=prefix, goal=goal, k=k,
                                              bsd=bsd, heuristic=heuristic):
                                args = make_args(graph, k, bsd)
                                args.graph_type = 'cube'
                                args.cube_buffer_dim = dim
                                path, _ = MODULES['XA'].XA(graph, prefix, goal, heuristic, False, args)
                                path = path or []
                                self.assertEqual(len(path), len(expected))
                                if path:
                                    self.assertEqual(path[:len(prefix)], prefix)
                                    self.assertEqual(path[-1], goal)
                                    self.assertEqual(len(path), len(set(path)))
                                    self.assertTrue(all(graph.has_edge(u, v) for u, v in zip(path, path[1:])))
                                    self.assertLessEqual(sum((u ^ v) == (1 << dim)
                                                             for u, v in zip(path, path[1:])), 1)

    def test_xa_rejects_a_prefix_with_two_buffer_crossings_before_expanding(self):
        graph = nx.Graph((v, v ^ (1 << d)) for v in range(8) for d in range(3))
        args = make_args(graph)
        args.graph_type = 'cube'
        args.cube_buffer_dim = 0
        with patch.object(State, 'generate_successors',
                          side_effect=AssertionError('Invalid prefix expanded')):
            path, stats = MODULES['XA'].XA(graph, [0, 1, 3, 2], 2, 'bcc_heuristic', False, args)
        self.assertIsNone(path)
        self.assertEqual(stats['expansions'], 0)

    def test_xa_preserves_buffer_history_on_snake_paths(self):
        vertices = [0, 1, 3, 7, 6]
        graph = nx.Graph(zip(vertices, vertices[1:]))
        for prefix in ([0], [0, 1]):
            for goal in (7, 6):
                for k in (1, 2, 3, 4):
                    with self.subTest(prefix=prefix, goal=goal, k=k):
                        args = make_args(graph, k)
                        args.graph_type = 'cube'
                        args.cube_buffer_dim = 0
                        path, _ = MODULES['XA'].XA(graph, prefix, goal, 'bcc_heuristic', True, args)
                        self.assertEqual(path or [], vertices[:4] if goal == 7 else [])


if __name__ == '__main__':
    unittest.main()
