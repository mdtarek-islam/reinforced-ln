import random

import pytest

torch = pytest.importorskip("torch")

from rl_recommender import RecommendationAgent, RLRecommender  # noqa: E402

EMB = 8
N = 12


def make_agent(tmp_path, **kwargs):
    torch.manual_seed(0)
    random.seed(0)
    data = torch.randn(N, EMB)
    return RecommendationAgent(EMB, data, stats_path=str(tmp_path / "stats.pth"), **kwargs)


def test_network_layers_match_spec():
    model = RLRecommender(384)
    assert (model.fc1.in_features, model.fc1.out_features) == (768, 128)
    assert (model.fc2.in_features, model.fc2.out_features) == (128, 64)
    assert (model.fc3.in_features, model.fc3.out_features) == (64, 1)
    assert model(torch.zeros(5, 768)).shape == (5, 1)
    assert all(p.requires_grad for p in model.parameters())


def test_greedy_selection_matches_score_ranking(tmp_path):
    agent = make_agent(tmp_path, epsilon=0.0)
    user = torch.randn(EMB)
    scores = agent.score_candidates(user)
    expected = torch.argsort(scores, descending=True)[:3].tolist()
    assert agent.select_actions(user) == expected


def test_select_actions_handles_single_item(tmp_path):
    agent = RecommendationAgent(EMB, torch.randn(1, EMB), epsilon=0.0, stats_path=str(tmp_path / "s.pth"))
    assert agent.select_actions(torch.randn(EMB), top_k=1) == [0]


def test_positive_feedback_raises_score_and_lowers_loss(tmp_path):
    agent = make_agent(tmp_path, epsilon=0.0, eval_interval=0)
    user = torch.randn(EMB)
    before = agent.score_candidates(user)[3].item()
    losses = [agent.train(user, 3, 1) for _ in range(30)]
    after = agent.score_candidates(user)[3].item()
    assert after > before
    assert losses[-1] < losses[0]
    assert agent.rewards_history == [1] * 30 and agent.training_steps == 30


def test_checkpoint_saved_every_interval_and_reloads(tmp_path):
    agent = make_agent(tmp_path, eval_interval=3)
    user = torch.randn(EMB)
    for reward in (1, -1, 1):
        agent.train(user, 0, reward)
    path = tmp_path / "stats.pth"
    assert path.exists()

    other = make_agent(tmp_path)
    checkpoint = other.load_model_weights(str(path))
    assert checkpoint["rewards_history"] == [1, -1, 1]
    assert checkpoint["training_steps"] == 3
    for a, b in zip(agent.model.parameters(), other.model.parameters()):
        assert torch.equal(a, b)
