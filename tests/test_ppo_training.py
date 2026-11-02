"""Smoke test of the whole offline PPO pipeline with a fake encoder and a tiny catalogue."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

import ppo_training  # noqa: E402
from ppo_agent import PPOAgent, PPOConfig  # noqa: E402


class FakeEncoder:
    """Hash each text to a stable random vector; genre words shift the vector so genres cluster."""
    def encode(self, texts, **kwargs):
        out = []
        for text in texts:
            vec = np.random.default_rng(abs(hash(text.split(" ")[0])) % (2 ** 32)).normal(size=16)
            out.append(vec + np.random.default_rng(len(text)).normal(0, 0.1, 16))
        return np.array(out)


def write_csv(path):
    genres = ["Action", "Romance", "Fantasy", "Drama"]
    rows = [{"ID": i, "Title": f"T{i}", "Genres": genres[i % 4], "Synopsis": f"{genres[i % 4]} story {i}"} for i in range(80)]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_experiment_writes_checkpoint_metrics_and_curve(tmp_path):
    csv = tmp_path / "n.csv"
    write_csv(csv)
    out = tmp_path / "models" / "ppo.pt"
    metrics = ppo_training.run_experiment(
        str(csv), str(out), updates=3, config=PPOConfig(k=2, slate_size=10, episodes_per_update=8, epochs=1, minibatch_size=8),
        seed=0, sim_threshold=-1.0, eval_episodes=20, cache_dir=str(tmp_path / "cache"),
        metrics_path=str(tmp_path / "docs" / "local" / "m.json"), curve_path=str(tmp_path / "docs" / "local" / "c.svg"),
        command="test", model=FakeEncoder(), log=lambda *_: None)
    assert out.exists() and (tmp_path / "docs" / "local" / "c.svg").exists()
    saved = json.loads((tmp_path / "docs" / "local" / "m.json").read_text())
    assert saved["label"] == "simulated feedback" and len(saved["mean_reward_per_update"]) == 3
    assert set(saved["final_mean_reward"]) >= {"ppo_greedy", "random_policy", "online_mlp_greedy"}
    assert metrics["rows"] == 80
    assert PPOAgent.from_checkpoint(str(out)).training_steps == 3
    assert list((tmp_path / "cache").glob("emb_*.npy"))


def test_embedding_cache_is_reused(tmp_path):
    csv = tmp_path / "n.csv"
    write_csv(csv)

    class Counting(FakeEncoder):
        calls = 0
        def encode(self, texts, **kwargs):
            Counting.calls += 1
            return super().encode(texts)

    ppo_training.load_catalogue(str(csv), str(tmp_path / "c"), model=Counting())
    ppo_training.load_catalogue(str(csv), str(tmp_path / "c"), model=Counting())
    assert Counting.calls == 1
