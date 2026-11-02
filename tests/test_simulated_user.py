import numpy as np
import pytest

from simulated_user import SimulatedUser, parse_genres

GENRES = ["Action, Fantasy", "Action", "Romance", "Romance, Drama", "Fantasy", "Drama"] * 5


def make_embeddings():
    """Embedding = genre indicator plus small noise, so genre and similarity agree."""
    rng = np.random.default_rng(0)
    names = ["Action", "Fantasy", "Romance", "Drama"]
    emb = np.array([[1.0 if n in parse_genres(g) else 0.0 for n in names] for g in GENRES]) + rng.normal(0, 0.05, (len(GENRES), 4))
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def test_parse_genres_handles_blank_and_nan():
    assert parse_genres("Action, Slice of Life") == frozenset({"Action", "Slice of Life"})
    assert parse_genres(float("nan")) == frozenset()
    assert parse_genres("") == frozenset()


def test_reward_needs_shared_genre_and_similarity():
    env = SimulatedUser(GENRES, make_embeddings(), sim_threshold=0.5, min_genre_support=3, seed=0)
    profile = env.sample_profile(np.random.default_rng(1))
    rewards = {env.reward(profile, i) for i in range(env.n_items)}
    assert rewards == {1.0, -1.0}
    for i in range(env.n_items):
        shares = bool(env.genre_sets[i] & profile.preferred)
        similar = float(env.user_embedding(profile) @ env.item_embeddings[i]) >= env.sim_threshold
        assert env.reward(profile, i) == (1.0 if shares and similar else -1.0)


def test_similarity_gate_blocks_a_genre_match():
    env = SimulatedUser(GENRES, make_embeddings(), sim_threshold=0.999, min_genre_support=3, seed=0)
    profile = env.sample_profile(np.random.default_rng(1))
    assert all(env.reward(profile, i) == -1.0 for i in range(env.n_items))


def test_profiles_are_deterministic_for_a_seeded_rng():
    env = SimulatedUser(GENRES, make_embeddings(), min_genre_support=3)
    a = env.sample_profile(np.random.default_rng(5))
    b = env.sample_profile(np.random.default_rng(5))
    assert a.preferred == b.preferred and np.allclose(a.user_embedding, b.user_embedding)


def test_group_ids_match_genre_strings():
    env = SimulatedUser(GENRES, make_embeddings(), min_genre_support=3)
    assert env.group_ids[0] == env.group_ids[6] != env.group_ids[1]


def test_rejects_when_no_genre_has_support():
    with pytest.raises(ValueError):
        SimulatedUser(GENRES, make_embeddings(), min_genre_support=999)
