import os

import numpy as np
import pandas as pd
import pytest

import data_loader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUCED_CSV = os.path.join(ROOT, "reduced_light_novels.csv")
SYNOPSES = os.path.join(ROOT, "synopses.txt")
COLUMNS = ["ID", "Title", "Medium_Picture", "Large_Picture", "Mean", "Rank", "Popularity", "Genres", "Synopsis"]


class FakeEncoder:
    """Stand-in for SBERT: one 4-dimensional vector per text."""

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(t), 1.0, 0.0, -1.0] for t in texts], dtype=np.float32)


def test_import_does_not_load_sbert():
    assert data_loader._sbert_model is None


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        data_loader.no_such_name


def test_load_dataset_on_real_csv_slice():
    df = data_loader.load_dataset(REDUCED_CSV, nrows=50)
    assert len(df) == 50
    assert list(df.columns[: len(COLUMNS)]) == COLUMNS
    assert df["Text_Representation"].map(type).eq(str).all()


def test_text_representation_handles_missing_values():
    df = pd.DataFrame({"Genres": ["Fantasy", None, None], "Synopsis": ["A hero.", "A villain.", None]})
    assert data_loader.build_text_representation(df).tolist() == ["Fantasy A hero.", " A villain.", " "]


def test_load_and_process_data_uses_given_model():
    encoder = FakeEncoder()
    df, embeddings = data_loader.load_and_process_data(REDUCED_CSV, model=encoder, nrows=20)
    assert len(encoder.calls) == 1
    assert encoder.calls[0] == df["Text_Representation"].tolist()
    assert np.asarray(embeddings).shape == (20, 4)


def test_load_synopses_reads_every_entry():
    synopses = data_loader.load_synopses(SYNOPSES)
    assert len(synopses) == 10
    assert synopses[0].startswith("In a futuristic metropolis")
    assert all("\n" not in s and not s.startswith("- ") for s in synopses)


def test_load_synopses_joins_wrapped_lines(tmp_path):
    path = tmp_path / "s.txt"
    path.write_text("- first line\ncontinues here\n\n- second\n", encoding="utf-8")
    assert data_loader.load_synopses(str(path)) == ["first line continues here", "second"]
