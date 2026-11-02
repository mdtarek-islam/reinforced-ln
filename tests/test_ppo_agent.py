"""Tests for the PPO actor-critic. They need PyTorch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppo_agent import (  # noqa: E402
    ActorCritic, PPOAgent, PPOConfig, clipped_surrogate_loss, collect_rollouts, compute_gae,
)

EMB = 4


class BanditEnv:
    """Fixed user, six items, reward +1 only for item 0."""
    n_items = 6
    item_embeddings = np.eye(6, EMB, dtype=np.float32) + 0.1
    group_ids = np.arange(6)

    def sample_profile(self, rng):
        return {"user": np.ones(EMB, dtype=np.float32)}

    def user_embedding(self, profile):
        return profile["user"]

    def reward(self, profile, idx):
        return 1.0 if idx == 0 else -1.0


def test_gae_matches_hand_computed_example():
    # delta2 = 2 - 0.25 = 1.75, delta1 = 0 + 0.9*0.25 - 1 = -0.775, delta0 = 1 + 0.9*1 - 0.5 = 1.4
    # A2 = 1.75, A1 = -0.775 + 0.45*1.75 = 0.0125, A0 = 1.4 + 0.45*0.0125 = 1.405625
    adv, ret = compute_gae(torch.tensor([1.0, 0.0, 2.0]), torch.tensor([0.5, 1.0, 0.25]), gamma=0.9, lam=0.5)
    assert torch.allclose(adv, torch.tensor([1.405625, 0.0125, 1.75]), atol=1e-6)
    assert torch.allclose(ret, torch.tensor([1.905625, 1.0125, 2.0]), atol=1e-6)


def test_gae_handles_batched_episodes_columnwise():
    r = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.zeros(2, 2)
    adv, _ = compute_gae(r, v, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[1.0, 1.0], [0.0, 1.0]]))


@pytest.mark.parametrize("adv, ratio, expect_sign", [
    (1.0, 1.0, -1),    # positive advantage, unclipped: loss falls as the ratio rises
    (1.0, 1.5, 0),     # positive advantage above 1+eps: clipped, no gradient
    (-1.0, 1.0, 1),    # negative advantage, unclipped: loss rises with the ratio
    (-1.0, 0.5, 0),    # negative advantage below 1-eps: clipped, no gradient
])
def test_clipped_objective_gradient_direction(adv, ratio, expect_sign):
    logp_old = torch.zeros(1)
    logp_new = torch.log(torch.tensor([ratio])).requires_grad_(True)
    loss = clipped_surrogate_loss(logp_new, logp_old, torch.tensor([adv]), clip_eps=0.2)
    loss.backward()
    grad = logp_new.grad.item()
    assert (grad > 0) - (grad < 0) == expect_sign


def test_policy_probabilities_sum_to_one_and_respect_mask():
    torch.manual_seed(0)
    net = ActorCritic(EMB)
    user = torch.randn(3, EMB)
    cand = torch.randn(3, 7, EMB)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[:, 2] = False
    probs = net.policy_probs(user, cand, mask)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.all(probs[:, 2] == 0)


def test_network_trunk_sizes_follow_the_prototype():
    net = ActorCritic(384)
    assert (net.fc1.in_features, net.fc1.out_features) == (768, 128)
    assert (net.fc2.in_features, net.fc2.out_features) == (128, 64)
    assert net.actor.out_features == 1 and net.critic.out_features == 1


def expected_reward(agent, env):
    user = torch.tensor(env.user_embedding(env.sample_profile(None))).unsqueeze(0)
    cand = torch.tensor(env.item_embeddings).unsqueeze(0)
    mask = torch.ones(1, env.n_items, dtype=torch.bool)
    with torch.no_grad():
        p = agent.net.policy_probs(user, cand, mask)[0]
    return (2 * p[0] - 1).item()


def test_one_ppo_update_improves_expected_reward_on_toy_problem():
    env = BanditEnv()
    cfg = PPOConfig(k=1, slate_size=6, episodes_per_update=256, epochs=8, minibatch_size=64, lr=3e-3,
                    diversity_penalty=0.0, entropy_coef=0.0)
    agent = PPOAgent(EMB, cfg, seed=0)
    before = expected_reward(agent, env)
    batch, _ = collect_rollouts(agent, env, np.random.default_rng(0))
    agent.update(batch)
    assert expected_reward(agent, env) > before


def test_checkpoint_round_trip_with_weights_only(tmp_path):
    agent = PPOAgent(EMB, PPOConfig(), seed=1)
    agent.rewards_history = [0.1, 0.2]
    agent.loss_history = [1.0, 0.5]
    path = str(tmp_path / "ppo.pt")
    agent.save(path)
    other = PPOAgent(EMB, PPOConfig(), seed=2)
    ckpt = other.load(path)
    assert ckpt["rewards_history"] == [0.1, 0.2]
    for a, b in zip(agent.net.parameters(), other.net.parameters()):
        assert torch.equal(a, b)
    # the file must load under the safe loader on its own
    assert "model_state_dict" in torch.load(path, map_location="cpu", weights_only=True)


def test_training_is_seedable():
    def run():
        env = BanditEnv()
        agent = PPOAgent(EMB, PPOConfig(k=2, slate_size=6, episodes_per_update=16, epochs=2, minibatch_size=16), seed=3)
        batch, _ = collect_rollouts(agent, env, np.random.default_rng(3))
        return agent.update(batch)["policy_loss"]
    assert run() == run()


def test_rollout_never_repeats_an_item_in_one_episode():
    env = BanditEnv()
    agent = PPOAgent(EMB, PPOConfig(k=4, slate_size=6, episodes_per_update=32), seed=0)
    batch, _ = collect_rollouts(agent, env, np.random.default_rng(0))
    picks = batch["actions"].reshape(4, 32)
    for e in range(32):
        assert len(set(picks[:, e].tolist())) == 4
