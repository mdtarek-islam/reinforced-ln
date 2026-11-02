"""Torch-free decision logic for the ReinforcedLN agent.

This module holds the parts of the recommendation policy that do not need a
neural network: the feedback reward scheme, the diversity penalty and the
epsilon-greedy choice between exploration and exploitation. The torch-based
``RecommendationAgent`` in ``rl_recommender.py`` calls these functions, so the
policy can be tested with plain NumPy arrays.
"""
import random

import numpy as np

# Number of most recent recommendations that receive the diversity penalty.
DIVERSITY_WINDOW = 5

POSITIVE_FEEDBACK = "Good"
NEGATIVE_FEEDBACK = "Bad"


def feedback_to_reward(feedback):
    """Map a feedback label to the reward used for training.

    The prototype uses a simple scheme: +1 for "Good" and -1 for anything else.
    """
    return 1 if feedback == POSITIVE_FEEDBACK else -1


def apply_diversity_penalty(scores, history, diversity_penalty, window=DIVERSITY_WINDOW):
    """Return a copy of ``scores`` with recent recommendations penalised.

    Each index in the last ``window`` entries of ``history`` loses
    ``diversity_penalty``. An index that appears twice in the window is
    penalised twice, which matches the original loop.
    """
    penalised = np.array(scores, dtype=np.float64, copy=True).reshape(-1)
    if window <= 0:
        return penalised
    for idx in history[-window:]:
        penalised[idx] -= diversity_penalty
    return penalised


def rank_top_k(scores, top_k):
    """Return the indices of the ``top_k`` highest scores, best first.

    Ties keep their original order, so the ranking is deterministic.
    """
    scores = np.asarray(scores).reshape(-1)
    _check_top_k(top_k, scores.shape[0])
    order = np.argsort(-scores, kind="stable")
    return order[:top_k].tolist()


def select_indices(scores, history, top_k=3, epsilon=0.1, diversity_penalty=0.2, rng=random):
    """Choose ``top_k`` candidate indices with an epsilon-greedy policy.

    With probability ``epsilon`` the choice is ``top_k`` random distinct
    indices (exploration). Otherwise the choice is the ``top_k`` best scores
    after the diversity penalty (exploitation). ``rng`` must offer
    ``uniform`` and ``sample``, as the ``random`` module does.
    """
    penalised = apply_diversity_penalty(scores, history, diversity_penalty)
    _check_top_k(top_k, penalised.shape[0])
    if rng.uniform(0, 1) < epsilon:
        return rng.sample(range(penalised.shape[0]), top_k)
    return rank_top_k(penalised, top_k)


def _check_top_k(top_k, num_items):
    if top_k < 1 or top_k > num_items:
        raise ValueError(f"top_k must be between 1 and {num_items}, got {top_k}")
