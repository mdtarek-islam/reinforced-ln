"""Load the reduced light novel dataset and compute SBERT embeddings.

The SBERT model ("all-MiniLM-L6-v2") loads on first use, not at import time.
``from data_loader import sbert_model`` still works: the module-level
``__getattr__`` builds the model when that name is first requested.
"""
import pandas as pd

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DATA_PATH = "reduced_light_novels.csv"

_sbert_model = None


def get_sbert_model():
    """Return the shared SBERT model, loading it on the first call."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    return _sbert_model


def __getattr__(name):
    # Lazy module attribute (PEP 562) so importing this module stays cheap.
    if name == "sbert_model":
        return get_sbert_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_text_representation(df):
    """Combine 'Genres' and 'Synopsis' into one string per novel.

    Missing values become empty strings, so every row yields a string.
    """
    return df["Genres"].fillna("") + " " + df["Synopsis"].fillna("")


def load_dataset(file_path=DEFAULT_DATA_PATH, nrows=None):
    """Read the dataset CSV and add the 'Text_Representation' column."""
    df = pd.read_csv(file_path, nrows=nrows)
    df["Text_Representation"] = build_text_representation(df)
    return df


def load_and_process_data(file_path=DEFAULT_DATA_PATH, model=None, nrows=None):
    """
    Loads and preprocesses the reduced light novel dataset.
    Combines the 'Genres' and 'Synopsis' fields into a single text representation
    and computes SBERT embeddings for each novel.

    ``model`` is any object with an ``encode(list_of_str)`` method. It defaults
    to the shared SBERT model. Returns ``(df, embeddings)``.
    """
    df = load_dataset(file_path, nrows=nrows)
    model = model if model is not None else get_sbert_model()

    # Compute SBERT Embeddings (one vector per row)
    embeddings = model.encode(df["Text_Representation"].tolist())

    return df, embeddings


def load_synopses(file_path="synopses.txt"):
    """Read test synopses from a text file.

    Each synopsis starts with "- " on a new line and may wrap over several
    lines. Wrapped lines are joined with single spaces.
    """
    with open(file_path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    synopses = []
    current = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                synopses.append(" ".join(current))
            current = [stripped[2:].strip()]
        elif stripped and current:
            current.append(stripped)
    if current:
        synopses.append(" ".join(current))
    return synopses


if __name__ == "__main__":
    df, embeddings = load_and_process_data()
    print(f"Loaded {len(df)} light novels with embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
