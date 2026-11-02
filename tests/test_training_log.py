import training_log


def test_record_and_reset():
    training_log.reset()
    synopses = training_log.sbert_synopsis_list
    assert training_log.record("story one", [0.1]) == 1
    assert training_log.record("story two", [0.2]) == 2
    assert training_log.training_attempts == 2
    assert training_log.sbert_synopsis_list == ["story one", "story two"]
    assert training_log.sbert_embedding_list == [[0.1], [0.2]]
    training_log.reset()
    assert training_log.training_attempts == 0
    assert synopses is training_log.sbert_synopsis_list and synopses == []
    assert training_log.sbert_embedding_list == []
