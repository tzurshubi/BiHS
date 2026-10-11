"""Tests for the touched-nodes counter (stats['touched']).

Run from the repository root with:
    PYTHONPATH=src python3 -m unittest src.tests.test_touched_nodes
"""
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from heuristics import heuristic as heuristic_module
from utils.utils import init_search_stats


NAMES = ('XA', 'XDFBnB', 'XIDA', 'XMM', 'BiXA', 'BiXDFBnB', 'BiXIDA')
MODULES = {name: importlib.import_module('algorithms.' + name) for name in NAMES}


def make_args(graph, lookahead):
    return SimpleNamespace(
        stats=init_search_stats(max(graph.nodes)), logger=lambda *a, **kw: None,
        graph_type='grid', cube_buffer_dim=None, lookahead=lookahead,
        bsd=False, sym_coil=False, solution_vertices=[], graph=graph,
        original_graph=graph, algo='XMM', backward_sym_generation=False,
        gui_logger=SimpleNamespace(video=False, gui_log_expansion=lambda *a: None),
    )


class TouchedNodesTests(unittest.TestCase):
    def test_touched_counts_every_heuristic_evaluation(self):
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4))
        for name in NAMES:
            for lookahead in (1, 3):
                with self.subTest(name=name, lookahead=lookahead):
                    args = make_args(graph, lookahead)
                    with patch.object(MODULES[name], 'heuristic', side_effect=heuristic_module.heuristic) as spy:
                        stats = getattr(MODULES[name], name)(graph, 0, 11, 'bcc_heuristic', False, args)[1]
                    self.assertEqual(stats['touched'], spy.call_count)
                    self.assertGreaterEqual(stats['touched'], stats['expansions'])

    def test_lookahead_touches_nodes_it_never_expands(self):
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4))
        stats = MODULES['XDFBnB'].XDFBnB(graph, 0, 11, 'bcc_heuristic', False, make_args(graph, 3))[1]
        self.assertGreater(stats['touched'], stats['expansions'])


if __name__ == '__main__':
    unittest.main()
