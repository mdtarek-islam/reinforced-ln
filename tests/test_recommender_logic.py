import random

import numpy as np
import pytest

from recommender_logic import (
    DIVERSITY_WINDOW,
    apply_diversity_penalty,
    feedback_to_reward,
    rank_top_k,
    select_indices,
)


@pytest.mark.parametrize("feedback, reward", [("Good", 1), ("Bad", -1), ("", -1), (None, -1)])
def test_feedback_to_reward(feedback, reward):
    assert feedback_to_reward(feedback) == reward


def test_penalty_hits_only_the_last_window():
    scores = np.zeros(10)
    history = [0, 1, 2, 3, 4, 5, 6]
    out = apply_diversity_penalty(scores, history, 0.2)
    assert DIVERSITY_WINDOW == 5
    assert out[0] == 0 and out[1] == 0
    assert np.allclose(out[2:7], -0.2)
    assert np.allclose(out[7:], 0)


def test_penalty_repeats_accumulate_and_input_is_unchanged():
    scores = np.ones(3)
    out = apply_diversity_penalty(scores, [1, 1], 0.5)
    assert out.tolist() == [1.0, 0.0, 1.0]
    assert scores.tolist() == [1.0, 1.0, 1.0]


def test_rank_top_k_orders_best_first_and_breaks_ties_by_index():
    assert rank_top_k([0.1, 0.9, 0.5, 0.9], 3) == [1, 3, 2]


@pytest.mark.parametrize("k", [0, 5])
def test_rank_top_k_rejects_bad_k(k):
    with pytest.raises(ValueError):
        rank_top_k([1, 2, 3, 4], k)


def test_greedy_selection_respects_diversity_penalty():
    scores = [0.5, 0.45, 0.1, 0.0]
    assert select_indices(scores, [], top_k=1, epsilon=0.0) == [0]
    assert select_indices(scores, [0], top_k=1, epsilon=0.0, diversity_penalty=0.2) == [1]


def test_exploration_returns_distinct_random_indices():
    rng = random.Random(7)
    picks = select_indices(np.arange(20.0), [], top_k=3, epsilon=1.0, rng=rng)
    assert len(picks) == 3 and len(set(picks)) == 3
    assert all(0 <= p < 20 for p in picks)


def test_epsilon_controls_exploration_rate():
    rng = random.Random(0)
    scores = np.arange(50.0)
    greedy = rank_top_k(scores, 3)
    runs = [select_indices(scores, [], top_k=3, epsilon=0.3, rng=rng) for _ in range(2000)]
    explore_rate = sum(r != greedy for r in runs) / len(runs)
    assert 0.25 < explore_rate < 0.35
