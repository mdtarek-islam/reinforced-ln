"""Shared in-memory log of user synopses entered during a session.

``ui_feedback.py`` updates these module-level values on each recommendation
request. Use ``record`` to add an entry and ``reset`` to clear the log.
"""
training_attempts = 0
sbert_synopsis_list = []
sbert_embedding_list = []


def record(synopsis, embedding):
    """Add one synopsis and its embedding, and count the attempt."""
    global training_attempts
    training_attempts += 1
    sbert_synopsis_list.append(synopsis)
    sbert_embedding_list.append(embedding)
    return training_attempts


def reset():
    """Clear the log in place so existing references stay valid."""
    global training_attempts
    training_attempts = 0
    sbert_synopsis_list.clear()
    sbert_embedding_list.clear()
