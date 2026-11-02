"""Simulated user for offline training (labelled "simulated feedback").

Each episode draws a hidden preference profile: one or two genres from the
catalogue genre set. The user embedding is the normalised mean of a few
novels that carry a preferred genre, so it behaves like the embedding of a
synopsis the user would write. The agent sees only that embedding.

Reward for one recommended novel:
  +1 if the novel shares at least one preferred genre AND its cosine
     similarity to the user embedding reaches ``sim_threshold``;
  -1 otherwise.

This module needs NumPy only.
"""
from dataclasses import dataclass

import numpy as np


def parse_genres(value):
    """Split a comma-separated genre string into a frozenset. Missing data gives an empty set."""
    if not isinstance(value, str):
        return frozenset()
    return frozenset(part.strip() for part in value.split(",") if part.strip())


@dataclass
class Profile:
    preferred: frozenset
    user_embedding: np.ndarray


class SimulatedUser:
    """Genre-overlap oracle with a synopsis-similarity gate.

    ``item_embeddings`` must be L2-normalised rows, so the dot product is the cosine.
    """

    def __init__(self, genre_strings, item_embeddings, sim_threshold=0.3, n_seed_items=5,
                 max_preferred=2, min_genre_support=10, seed=0):
        self.genre_sets = [parse_genres(g) for g in genre_strings]
        self.item_embeddings = np.asarray(item_embeddings, dtype=np.float32)
        self.n_items = len(self.genre_sets)
        self.sim_threshold = sim_threshold
        self.n_seed_items = n_seed_items
        self.max_preferred = max_preferred

        counts = {}
        for genres in self.genre_sets:
            for genre in genres:
                counts[genre] = counts.get(genre, 0) + 1
        self.genre_pool = sorted(g for g, c in counts.items() if c >= min_genre_support)
        if not self.genre_pool:
            raise ValueError(f"no genre has at least {min_genre_support} novels")
        self.items_by_genre = {g: [i for i, s in enumerate(self.genre_sets) if g in s] for g in self.genre_pool}

        # One group id per distinct genre string, used for the diversity penalty.
        ids = {}
        self.group_ids = np.array([ids.setdefault(s, len(ids)) for s in genre_strings], dtype=np.int64)

    def sample_profile(self, rng):
        """Draw a hidden profile using the numpy ``Generator`` ``rng``."""
        n_pref = int(rng.integers(1, self.max_preferred + 1))
        n_pref = min(n_pref, len(self.genre_pool))
        chosen = rng.choice(len(self.genre_pool), size=n_pref, replace=False)
        preferred = frozenset(self.genre_pool[i] for i in chosen)
        pool = sorted({i for g in preferred for i in self.items_by_genre[g]})
        seeds = rng.choice(pool, size=min(self.n_seed_items, len(pool)), replace=False)
        mean = self.item_embeddings[seeds].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return Profile(preferred, (mean / norm if norm > 0 else mean).astype(np.float32))

    def user_embedding(self, profile):
        return profile.user_embedding

    def similarity(self, profile, idx):
        return float(profile.user_embedding @ self.item_embeddings[idx])

    def reward(self, profile, idx):
        """Return +1.0 or -1.0 for recommending novel ``idx``."""
        shares = bool(self.genre_sets[idx] & profile.preferred)
        return 1.0 if shares and self.similarity(profile, idx) >= self.sim_threshold else -1.0
