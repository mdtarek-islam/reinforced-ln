import os

import pandas as pd

from reducer import has_synopsis, reduce_dataset, unique_genres

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_df(n=200):
    genres = ["Action, Fantasy", "Romance", "Comedy, Romance", "Fantasy", None]
    rows = []
    for i in range(n):
        rows.append({
            "ID": i,
            "Title": f"Novel {i}",
            "Genres": genres[i % len(genres)],
            "Synopsis": f"Story {i}" if i % 3 else None,
        })
    rows.append(dict(rows[0]))  # duplicate ID
    return pd.DataFrame(rows)


def test_unique_genres_is_sorted_and_skips_blanks():
    assert unique_genres(pd.Series(["B, A", None, "", "C"])) == ["A", "B", "C"]


def test_has_synopsis_mask():
    df = pd.DataFrame({"Synopsis": ["x", "  ", None]})
    assert has_synopsis(df).tolist() == [True, False, False]


def test_output_has_unique_ids_and_respects_target():
    out = reduce_dataset(make_df(), target_size=60)
    assert len(out) == 60
    assert out["ID"].is_unique


def test_multi_genre_novels_are_not_duplicated():
    # Every novel has two genres, so per-genre sampling draws each one twice.
    df = pd.DataFrame({
        "ID": range(10),
        "Title": [f"N{i}" for i in range(10)],
        "Genres": ["Action, Drama"] * 10,
        "Synopsis": ["text"] * 10,
    })
    out = reduce_dataset(df, target_size=8)
    assert out["ID"].is_unique


def test_result_is_deterministic():
    a = reduce_dataset(make_df(), target_size=50)
    b = reduce_dataset(make_df(), target_size=50)
    pd.testing.assert_frame_equal(a, b)


def test_no_genres_falls_back_to_synopsis_pool():
    df = pd.DataFrame({"ID": [1, 2, 3], "Title": list("abc"), "Genres": [None] * 3, "Synopsis": ["x", None, "y"]})
    out = reduce_dataset(df, target_size=5)
    assert sorted(out["ID"]) == [1, 3]


def test_real_dataset_slice():
    df = pd.read_csv(os.path.join(ROOT, "light_novels.csv"), nrows=1500)
    out = reduce_dataset(df, target_size=200)
    assert len(out) == 200
    assert out["ID"].is_unique
    assert set(out["ID"]).issubset(set(df["ID"]))
