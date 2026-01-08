#!/usr/bin/env python3
"""
Reinforcement Learning for Snake-in-the-Box traversal on a d-dimensional hypercube.

Goal:
- Start at vertex 0 in Q_d (vertices 0..2^d-1).
- Build a *snake* (an induced path): when moving to a new vertex v (neighbor of current head),
  v must NOT be adjacent to any earlier vertex in the path except the current head.
- Conceptually, illegal vertices are "deleted" as the path grows. We implement this efficiently
  by maintaining a visited bitset and checking adjacency constraints.

RL:
- Episodic Q-learning with linear function approximation.
- Actions are dimensions to flip: a in {0..d-1}.
- Reward = +1 per legal move (maximize path length).
- Parameters (weights) are saved to disk and loaded on rerun.

Usage examples:
  python snake_rl_hypercube.py --d 8 --K 5000
  python snake_rl_hypercube.py --d 10 --K 20000 --epsilon 0.2
  python snake_rl_hypercube.py --d 8 --eval-only --eval-episodes 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np


# ---------------------------
# Bitset helpers (Python int)
# ---------------------------

def bit_set(mask: int, i: int) -> int:
    return mask | (1 << i)

def bit_clear(mask: int, i: int) -> int:
    return mask & ~(1 << i)

def bit_has(mask: int, i: int) -> bool:
    return (mask >> i) & 1 == 1

def popcount(x: int) -> int:
    return x.bit_count()


# ---------------------------
# Hypercube + Snake constraint
# ---------------------------

@dataclass
class SnakeEnv:
    d: int

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be >= 1")
        self.n = 1 << self.d
        # neighbor_mask[v] is a bitset (Python int) of all neighbors of v.
        self.neighbor_mask = self._precompute_neighbor_masks()

    def _precompute_neighbor_masks(self) -> List[int]:
        masks = [0] * self.n
        for v in range(self.n):
            m = 0
            for dim in range(self.d):
                m |= 1 << (v ^ (1 << dim))
            masks[v] = m
        return masks

    def reset(self) -> "EpisodeState":
        # Path starts at 0. visited includes current head.
        return EpisodeState(
            head=0,
            step=0,
            visited_mask=(1 << 0),
            path=[0],
        )

    def legal_actions(self, st: "EpisodeState") -> List[int]:
        # Action a flips bit a => next vertex v = head ^ (1<<a)
        # Legal iff:
        # 1) v is unvisited
        # 2) v is NOT adjacent to any earlier vertex in path except the head.
        #    Equivalent: neighbors(v) ∩ (visited \ {head}) must be empty.
        visited_wo_head = bit_clear(st.visited_mask, st.head)

        head = st.head
        legal: List[int] = []
        for a in range(self.d):
            v = head ^ (1 << a)
            if bit_has(st.visited_mask, v):
                continue
            # induced path check
            if (self.neighbor_mask[v] & visited_wo_head) != 0:
                continue
            legal.append(a)
        return legal

    def step(self, st: "EpisodeState", action: int) -> Tuple["EpisodeState", float, bool]:
        # Returns (new_state, reward, done)
        head = st.head
        v = head ^ (1 << action)

        # Validate legality (defensive)
        legal = self.legal_actions(st)
        if action not in legal:
            # illegal: terminate with no reward (you can change this to negative reward if you want)
            return st, 0.0, True

        new_visited = bit_set(st.visited_mask, v)
        new_path = st.path + [v]
        new_state = EpisodeState(
            head=v,
            step=st.step + 1,
            visited_mask=new_visited,
            path=new_path,
        )
        reward = 1.0
        done = (len(self.legal_actions(new_state)) == 0)
        return new_state, reward, done


@dataclass
class EpisodeState:
    head: int
    step: int
    visited_mask: int
    path: List[int]


# ---------------------------
# Linear Q-learning agent
# ---------------------------

@dataclass
class LinearQAgent:
    d: int
    alpha: float = 0.05
    gamma: float = 0.99
    epsilon: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.n_actions = self.d

        # Feature vector: [bias, step_norm, legal_frac, head_bits(d), legal_bits(d)] => F = 3 + 2d
        self.F = 3 + 2 * self.d

        # Weight matrix W[a, f]
        self.W = np.zeros((self.n_actions, self.F), dtype=np.float64)

    def featurize(self, st: EpisodeState, legal_actions: List[int], n_vertices: int) -> np.ndarray:
        # step_norm: normalized by max possible vertices (n_vertices-1 edges)
        step_norm = st.step / max(1, (n_vertices - 1))
        legal_frac = len(legal_actions) / max(1, self.d)

        head_bits = np.array([(st.head >> i) & 1 for i in range(self.d)], dtype=np.float64)
        legal_bits = np.zeros(self.d, dtype=np.float64)
        for a in legal_actions:
            legal_bits[a] = 1.0

        phi = np.concatenate((
            np.array([1.0, step_norm, legal_frac], dtype=np.float64),
            head_bits,
            legal_bits,
        ))
        return phi

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        # Q(a) = W[a] dot phi
        return self.W @ phi

    def select_action(self, legal_actions: List[int], q: np.ndarray) -> int:
        # epsilon-greedy over legal actions only
        if not legal_actions:
            raise ValueError("No legal actions to select from")

        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # argmax among legal actions
        best_a = legal_actions[0]
        best_q = q[best_a]
        for a in legal_actions[1:]:
            if q[a] > best_q:
                best_q = q[a]
                best_a = a
        return best_a

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: Optional[np.ndarray],
        legal_next: Optional[List[int]],
    ) -> None:
        # TD target
        q_sa = float(self.W[action] @ phi)

        if phi_next is None or not legal_next:
            target = reward
        else:
            q_next_all = self.W @ phi_next
            # max over legal actions
            max_next = float(np.max(q_next_all[legal_next]))
            target = reward + self.gamma * max_next

        td = target - q_sa
        self.W[action] += self.alpha * td * phi

    def save(self, path: Path, meta: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W=self.W,
            meta=json.dumps(meta),
        )

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=False)
        W = data["W"]
        if W.shape != self.W.shape:
            raise ValueError(f"Saved weights shape {W.shape} != expected {self.W.shape}")
        self.W[:] = W
        return True


# ---------------------------
# Training / Evaluation
# ---------------------------

@dataclass
class RunStats:
    lengths: List[int]  # number of vertices in snake path
    best_path: List[int]

    def summary(self) -> Dict[str, float]:
        arr = np.array(self.lengths, dtype=np.float64)
        return {
            "episodes": float(len(self.lengths)),
            "min_len": float(np.min(arr)) if len(arr) else 0.0,
            "mean_len": float(np.mean(arr)) if len(arr) else 0.0,
            "median_len": float(np.median(arr)) if len(arr) else 0.0,
            "max_len": float(np.max(arr)) if len(arr) else 0.0,
            "std_len": float(np.std(arr)) if len(arr) else 0.0,
        }


def run_episodes(
    env: SnakeEnv,
    agent: LinearQAgent,
    K: int,
    train: bool,
    verbose_every: int = 0,
) -> RunStats:
    lengths: List[int] = []
    best_path: List[int] = []

    for ep in range(1, K + 1):
        st = env.reset()
        done = False

        while not done:
            legal = env.legal_actions(st)
            if not legal:
                done = True
                break

            phi = agent.featurize(st, legal, env.n)
            q = agent.q_values(phi)
            a = agent.select_action(legal, q)

            st_next, r, done = env.step(st, a)

            if train:
                legal_next = env.legal_actions(st_next) if not done else []
                phi_next = agent.featurize(st_next, legal_next, env.n) if not done else None
                agent.update(phi, a, r, phi_next, legal_next)

            st = st_next

        L_edges = len(st.path) - 1
        lengths.append(L_edges)
        if L_edges > len(best_path) - 1:
            best_path = st.path


        if verbose_every and (ep % verbose_every == 0):
            recent = lengths[-verbose_every:]
            print(
                f"[ep {ep}/{K}] "
                f"recent mean={np.mean(recent):.2f} "
                f"recent max={np.max(recent)} "
                f"best={len(best_path)-1}"
            )

    return RunStats(lengths=lengths, best_path=best_path)


def print_stats(title: str, stats: RunStats) -> None:
    s = stats.summary()
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Episodes: {int(s['episodes'])}")
    print(f"Path length (edges): min={int(s['min_len'])}, "
        f"median={s['median_len']:.1f}, mean={s['mean_len']:.2f}, "
        f"max={int(s['max_len'])}, std={s['std_len']:.2f}")
    print(f"Best path length (edges) = {len(stats.best_path) - 1}")
    print(f"Best path): {stats.best_path}")


def histogram(lengths: List[int], width: int = 50) -> None:
    if not lengths:
        return
    counts: Dict[int, int] = {}
    for L in lengths:
        counts[L] = counts.get(L, 0) + 1
    keys = sorted(counts.keys())
    max_c = max(counts.values())

    print("\nHistogram of path lengths (vertices):")
    for k in keys:
        c = counts[k]
        bar = "#" * int(round(width * (c / max_c))) if max_c else ""
        print(f"{k:4d}: {bar} ({c})")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    DEFAULT_D = 10
    DEFAULT_K = 50000000
    DEFAULT_ALPHA = 0.05
    DEFAULT_GAMMA = 0.99
    DEFAULT_EPSILON = 0.15
    DEFAULT_SEED = 0
    DEFAULT_MODEL_DIR = "models"
    DEFAULT_EVAL_ONLY = False
    DEFAULT_EVAL_EPISODES = 300
    DEFAULT_VERBOSE_EVERY = 10000
    DEFAULT_SAVE_BEST_PATH = ""

    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=DEFAULT_D, help="Hypercube dimension d")
    ap.add_argument("--K", type=int, default=DEFAULT_K, help="Training episodes")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount")
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Exploration rate")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory for saved params")
    ap.add_argument("--eval-only", default=DEFAULT_EVAL_ONLY, action="store_true", help="Skip training, just evaluate loaded model")
    ap.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="Evaluation episodes")
    ap.add_argument("--verbose-every", type=int, default=DEFAULT_VERBOSE_EVERY, help="Print progress every N episodes (0=off)")
    ap.add_argument("--save-best-path", type=str, default=DEFAULT_SAVE_BEST_PATH, help="Write best path to JSON file")
    args = ap.parse_args()

    env = SnakeEnv(d=args.d)

    agent = LinearQAgent(
        d=args.d,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    model_path = Path(args.model_dir) / f"snake_rl_Q{args.d}.npz"
    loaded = agent.load(model_path)

    if loaded:
        print(f"Loaded parameters from: {model_path}")
    else:
        print(f"No saved parameters found at: {model_path} (starting fresh)")

    if not args.eval_only:
        print("\nTraining...")
        train_stats = run_episodes(
            env=env,
            agent=agent,
            K=args.K,
            train=True,
            verbose_every=args.verbose_every,
        )
        print_stats("Training stats", train_stats)
        histogram(train_stats.lengths)

        meta = {
            "d": args.d,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "features": "bias, step_norm, legal_frac, head_bits(d), legal_bits(d)",
        }
        agent.save(model_path, meta)
        print(f"\nSaved parameters to: {model_path}")

    # Evaluation with epsilon forced to 0 (greedy)
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    print("\nEvaluating (greedy policy)...")
    eval_stats = run_episodes(
        env=env,
        agent=agent,
        K=args.eval_episodes,
        train=False,
        verbose_every=0,
    )
    print_stats("Evaluation stats", eval_stats)
    histogram(eval_stats.lengths)

    agent.epsilon = old_eps

    if args.save_best_path:
        out = {
            "d": args.d,
            "best_path_vertices": eval_stats.best_path,
            "best_path_edges": len(eval_stats.best_path) - 1,
        }
        Path(args.save_best_path).write_text(json.dumps(out, indent=2))
        print(f"\nWrote best path to: {args.save_best_path}")


if __name__ == "__main__":
    main()
