import math

import pandas as pd

from extractor import node_to_row
from ui_feedback import cover_url_from_record, fetch_cover_image, format_recommendation


def test_format_recommendation():
    rec = pd.Series({"Title": "T", "Genres": "Fantasy", "Synopsis": "S"})
    assert format_recommendation(rec) == "### T\n**Genres:** Fantasy\n\n**Synopsis:** S"


def test_cover_url_normalises_slashes():
    rec = pd.Series({"Medium_Picture": "https:\\\\cdn\\img.jpg"})
    assert cover_url_from_record(rec) == "https://cdn/img.jpg"


def test_cover_url_missing_values():
    for value in (math.nan, None, "", "N/A"):
        assert cover_url_from_record(pd.Series({"Medium_Picture": value}, dtype=object)) is None
    assert cover_url_from_record(pd.Series({"Title": "x"})) is None
    assert fetch_cover_image(None) is None


def test_node_to_row_handles_null_fields():
    row = node_to_row({"id": 5, "title": "X", "main_picture": None, "synopsis": None,
                       "genres": [{"name": "Action"}, {"name": "Drama"}]})
    assert row == [5, "X", "N/A", "N/A", "N/A", "N/A", "N/A", "Action, Drama", "N/A"]


def test_node_to_row_flattens_synopsis():
    row = node_to_row({"id": 1, "title": "Y", "main_picture": {"medium": "m", "large": "l"},
                       "synopsis": "a\nb", "mean": 8.1, "rank": 3, "popularity": 9})
    assert row == [1, "Y", "m", "l", 8.1, 3, 9, "", "a b"]
