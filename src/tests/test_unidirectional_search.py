"""Regression tests for expansion accounting and exact-goal detection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from algorithms.XA import XA
from algorithms.XDFBnB import XDFBnB
from models.heapq_state import HeapqState
from models.state import State
from utils.utils import init_search_stats


def make_args(graph, lookahead=2, bsd=False):
    return SimpleNamespace(
        stats=init_search_stats(max(graph.nodes)), logger=lambda *a, **kw: None,
        graph_type='grid', cube_buffer_dim=None, lookahead=lookahead,
        bsd=bsd, sym_coil=False, solution_vertices=[], graph=graph,
        original_graph=graph, gui_logger=SimpleNamespace(video=False),
    )


class UnidirectionalSearchTests(unittest.TestCase):
    def test_xa_does_not_count_the_terminating_pop(self):
        # Both branches initially tie. After one finds the optimum, the other
        # remains in OPEN and must be popped without generating successors.
        graph = nx.Graph([(0, 1), (1, 3), (0, 2), (2, 3)])
        args = make_args(graph, lookahead=1)
        popped, expanded = [], []
        original_pop = HeapqState.pop
        original_generate = State.generate_successors

        def pop(queue):
            entry = original_pop(queue)
            popped.append(entry[-1])
            return entry

        def generate(state, *a, **kw):
            expanded.append(state)
            return original_generate(state, *a, **kw)

        with patch.object(HeapqState, 'pop', pop), \
                patch.object(State, 'generate_successors', generate):
            path, stats = XA(graph, 0, 3, 'bcc_heuristic', False, args)

        self.assertEqual(len(path) - 1, 2)
        self.assertEqual(len(popped), 3)
        self.assertEqual(len(expanded), 2)
        self.assertNotIn(popped[-1], expanded)
        self.assertEqual(stats['expansions'], len(expanded))

    def test_xa_goal_root_requires_no_expansion(self):
        graph = nx.path_graph(1)
        with patch.object(State, 'generate_successors',
                          side_effect=AssertionError('Goal state expanded')):
            path, stats = XA(graph, 0, 0, 'bcc_heuristic', False, make_args(graph))
        self.assertEqual(path, [0])
        self.assertEqual(stats['expansions'], 0)

    def test_xdfbnb_generates_the_goal_at_lookahead_boundaries(self):
        for n in (2, 3, 4, 5, 6):
            for lookahead in (1, 2, 3, 4):
                for snake in (False, True):
                    for bsd in (False, True):
                        with self.subTest(n=n, lookahead=lookahead, snake=snake, bsd=bsd):
                            graph = nx.path_graph(n)
                            args = make_args(graph, lookahead, bsd)
                            generated_heads = set()
                            original_generate = State.generate_successors

                            def generate(state, *a, **kw):
                                successors = original_generate(state, *a, **kw)
                                generated_heads.update(s.head for s in successors)
                                return successors

                            with patch.object(State, 'generate_successors', generate):
                                path, _ = XDFBnB(graph, 0, n - 1, 'bcc_heuristic', snake, args)
                            self.assertEqual(path, list(range(n)))
                            self.assertIn(n - 1, generated_heads)


if __name__ == '__main__':
    unittest.main()
