"""Regression tests for the -1 heuristic sentinel.

Run from the repository root with:
    PYTHONPATH=src sage -python -m unittest discover -s src/tests -p 'test_heuristic_dead_end_pruning.py'
"""
import heapq
import importlib
import random
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from models.heapq_state import HeapqState
from models.openvopen import Openvopen
from models.state import State
from utils.utils import init_search_stats


NAMES = ('XA', 'XDFBnB', 'XIDA', 'XMM', 'BiXA', 'BiXDFBnB', 'BiXIDA')
MODULES = {name: importlib.import_module('algorithms.' + name) for name in NAMES}


def make_args(graph, lookahead=2, bsd=False, video=False):
    def gui_log_expansion(state, f, h, successors):
        assert h != -1, 'A dead state was logged as expanded'
        assert all(sh != -1 for _, _, sh in successors), 'A dead successor was logged'

    return SimpleNamespace(
        stats=init_search_stats(max(graph.nodes)), logger=lambda *a, **kw: None,
        graph_type='grid', cube_buffer_dim=None, lookahead=lookahead,
        bsd=bsd, sym_coil=False, solution_vertices=[], graph=graph,
        original_graph=graph, algo='XMM', backward_sym_generation=False,
        gui_logger=SimpleNamespace(video=video, gui_log_expansion=gui_log_expansion),
    )


def oracle(graph, start, goal, snake):
    if start == goal:
        return [start]
    paths = nx.all_simple_paths(graph, start, goal)
    if snake:
        paths = (p for p in paths if all(
            not graph.has_edge(p[i], p[j])
            for i in range(len(p)) for j in range(i + 2, len(p))
        ))
    return max(paths, key=len, default=[])


class DeadEndPruningTests(unittest.TestCase):
    def run_search(self, name, graph, start, goal, args, snake=False, heuristic='bcc_heuristic'):
        return getattr(MODULES[name], name)(graph, start, goal, heuristic, snake, args)

    def assert_path(self, graph, path, start, goal, snake, expected):
        path = path or []
        self.assertEqual(len(path), len(expected))
        if path:
            self.assertEqual((path[0], path[-1]), (start, goal))
            self.assertEqual(len(set(path)), len(path))
            self.assertTrue(all(graph.has_edge(a, b) for a, b in zip(path, path[1:])))
            if snake:
                self.assertTrue(all(not graph.has_edge(path[i], path[j])
                                    for i in range(len(path)) for j in range(i + 2, len(path))))

    def test_dead_initial_prefix_is_pruned_before_any_expansion(self):
        graph = nx.Graph([(0, 1), (1, 2), (4, 5)])
        for name in NAMES:
            for bsd in (False, True):
                with self.subTest(name=name, bsd=bsd):
                    args = make_args(graph, bsd=bsd)
                    with patch.object(State, 'generate_successors', side_effect=AssertionError('Dead root expanded')):
                        result = self.run_search(name, graph, [0, 1, 2], 5, args)
                    self.assertFalse(result[0])
                    self.assertEqual(args.stats['expansions'], 0)
                    self.assertEqual(args.stats['violations']['heuristic'][2], 1)

    def test_xmm_rejects_a_dead_backward_root(self):
        graph = nx.path_graph(5)
        args = make_args(graph)

        def h(state, *unused):
            return -1 if state.head == 4 else 4

        with patch.object(MODULES['XMM'], 'heuristic', side_effect=h), \
                patch.object(HeapqState, 'push', side_effect=AssertionError('Dead root entered OPEN')):
            self.assertFalse(self.run_search('XMM', graph, 0, 4, args)[0])
        self.assertEqual(args.stats['expansions'], 0)
        self.assertEqual(args.stats['violations']['heuristic'][0], 1)

    def test_ida_dead_leaves_do_not_schedule_another_iteration(self):
        graph = nx.Graph([(0, 1), (1, 2), (3, 4), (4, 5)])
        for name in ('XIDA', 'BiXIDA'):
            with self.subTest(name=name):
                args = make_args(graph, lookahead=1 if name == 'XIDA' else 2)

                def h(state, target, *unused):
                    g = state.g + (target.g if isinstance(target, State) else 0)
                    return -1 if g else len(graph)

                with patch.object(MODULES[name], 'heuristic', side_effect=h):
                    self.assertFalse(self.run_search(name, graph, 0, 5, args)[0])
                self.assertEqual(args.stats['expansions'], 1)
                self.assertGreater(sum(args.stats['violations']['heuristic'].values()), 0)

    def test_dead_successors_never_enter_open_or_expand(self):
        # Generate the dead branches first, before IDA can prove the optimum.
        graph = nx.Graph([(0, 5), (5, 6), (6, 7), (7, 8),
                          (4, 9), (9, 10), (10, 11), (11, 12),
                          (0, 1), (1, 2), (2, 3), (3, 4)])
        for name in NAMES:
            modes = (-2, -1, 1, 2, 3, 4) if name.startswith('Bi') else (1, 2, 3, 4)
            if name == 'XMM':
                modes = (2,)
            for k in modes:
                with self.subTest(name=name, lookahead=k):
                    args = make_args(graph, lookahead=k, video=True)
                    dead_states, dead_pairs, keep_alive = set(), set(), []
                    module = MODULES[name]
                    original_h = module.heuristic
                    original_generate = State.generate_successors
                    original_push = HeapqState.push
                    original_insert = Openvopen.insert_state
                    original_heappush = heapq.heappush

                    def h(state, target, *a, **kw):
                        value = original_h(state, target, *a, **kw)
                        if value == -1:
                            keep_alive.append((state, target))
                            if isinstance(target, State):
                                dead_pairs.add((id(state), id(target)))
                            else:
                                dead_states.add(id(state))
                        return value

                    def generate(state, *a, **kw):
                        self.assertNotIn(id(state), dead_states)
                        return original_generate(state, *a, **kw)

                    def push(queue, state, *a, **kw):
                        self.assertNotIn(id(state), dead_states)
                        return original_push(queue, state, *a, **kw)

                    def insert(index, state, *a, **kw):
                        self.assertNotIn(id(state), dead_states)
                        return original_insert(index, state, *a, **kw)

                    def heappush(queue, entry):
                        payload = entry[-1]
                        if isinstance(payload, tuple) and len(payload) == 4:
                            self.assertNotIn((id(payload[0]), id(payload[1])), dead_pairs)
                        return original_heappush(queue, entry)

                    def profile(frame, event, result):
                        if event == 'call' and frame.f_code.co_name in ('exp_n_check_states', 'get_lookahead_successors'):
                            local = frame.f_locals
                            f = local.get('state_F', local.get('cur_F'))
                            b = local.get('state_B', local.get('cur_B'))
                            if f is not None and b is not None:
                                self.assertNotIn((id(f), id(b)), dead_pairs)

                    previous_profile = sys.getprofile()
                    try:
                        with ExitStack() as stack:
                            stack.enter_context(patch.object(module, 'heuristic', side_effect=h))
                            stack.enter_context(patch.object(State, 'generate_successors', generate))
                            stack.enter_context(patch.object(HeapqState, 'push', push))
                            stack.enter_context(patch.object(Openvopen, 'insert_state', insert))
                            stack.enter_context(patch.object(heapq, 'heappush', heappush))
                            sys.setprofile(profile)
                            result = self.run_search(name, graph, 0, 4, args)
                    finally:
                        sys.setprofile(previous_profile)
                    self.assertTrue(dead_states or dead_pairs, 'Case did not exercise sentinel pruning')
                    self.assert_path(graph, result[0], 0, 4, False, [0, 1, 2, 3, 4])

    def test_zero_and_one_edge_solutions_are_kept(self):
        for name in NAMES:
            for n in (1, 2, 5):
                for snake in (False, True):
                    with self.subTest(name=name, n=n, snake=snake):
                        graph = nx.path_graph(n)
                        result = self.run_search(name, graph, 0, n - 1, make_args(graph), snake)
                        self.assertEqual(result[0], list(range(n)))

    def test_other_heuristic_modes_remain_usable(self):
        graph = nx.path_graph(5)
        for name in NAMES:
            for heuristic in (None, 'heuristic0', 'reachable_heuristic'):
                with self.subTest(name=name, heuristic=heuristic):
                    result = self.run_search(name, graph, 0, 4, make_args(graph), heuristic=heuristic)
                    self.assertEqual(result[0], [0, 1, 2, 3, 4])

    def test_small_graphs_match_exhaustive_search(self):
        rng = random.Random(1922)
        for i in range(40):
            n = rng.randrange(3, 8)
            graph = nx.gnp_random_graph(n, rng.uniform(.15, .65), seed=rng)
            expected = {s: oracle(graph, 0, n - 1, s) for s in (False, True)}
            for name in NAMES:
                for snake in (False, True):
                    for bsd in (False, True):
                        with self.subTest(graph=i, name=name, snake=snake, bsd=bsd):
                            args = make_args(graph, lookahead=1 + i % 4, bsd=bsd)
                            if name == 'XMM':
                                # The existing cutoff/AUXOPEN termination rule
                                # has independent optimality failures. Exercise
                                # its best-first mode here; the sentinel tests
                                # above exercise full XMM with the cutoff enabled.
                                args.algo = 'light'
                            result = self.run_search(name, graph, 0, n - 1, args, snake)
                            self.assert_path(graph, result[0], 0, n - 1, snake, expected[snake])


if __name__ == '__main__':
    unittest.main()
