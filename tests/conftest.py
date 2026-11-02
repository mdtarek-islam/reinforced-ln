"""Put the repo root and dataset_extraction/ on sys.path for the tests."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "dataset_extraction")):
    if path not in sys.path:
        sys.path.insert(0, path)
