"""Offline PPO training against the simulated user ("simulated feedback").

``run_experiment`` embeds the catalogue (cached), trains the PPO agent,
trains the old online MLP agent on the same simulated feedback, evaluates
both and a random policy on the same held-out episodes, then writes
``docs/local/repro-metrics.json`` and ``docs/local/ppo-training-curve.svg``.
"""
import hashlib
import json
import os
import platform
import random
import time

import numpy as np

from simulated_user import SimulatedUser

DEFAULT_CACHE_DIR = "embeddings_cache"
DEFAULT_METRICS_PATH = os.path.join("docs", "local", "repro-metrics.json")
DEFAULT_CURVE_PATH = os.path.join("docs", "local", "ppo-training-curve.svg")
DEFAULT_MODELS_ENV = "REINFORCEDLN_MODELS_DIR"
DEFAULT_MODELS_DIR = "models"


def models_dir():
    """Weight folder: the env var if set, else a gitignored ``models/`` folder in the repo."""
    return os.environ.get(DEFAULT_MODELS_ENV) or DEFAULT_MODELS_DIR


def load_catalogue(csv_path, cache_dir=DEFAULT_CACHE_DIR, model=None, model_name=None):
    """Return ``(df, embeddings)`` with L2-normalised float32 embeddings, cached on disk.

    The cache key hashes the model name and every text, so a changed CSV re-embeds.
    """
    from data_loader import SBERT_MODEL_NAME, get_sbert_model, load_dataset

    df = load_dataset(csv_path)
    texts = df["Text_Representation"].tolist()
    name = model_name or SBERT_MODEL_NAME
    digest = hashlib.sha256((name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"emb_{digest}.npy")
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
    else:
        model = model if model is not None else get_sbert_model()
        emb = np.asarray(model.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, emb)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return df, (emb / np.clip(norms, 1e-12, None)).astype(np.float32)


def build_env(df, embeddings, sim_threshold, seed):
    return SimulatedUser(df["Genres"].tolist(), embeddings, sim_threshold=sim_threshold, seed=seed)


def make_eval_set(env, n_episodes, slate_size, seed):
    """Fixed held-out profiles and slates shared by every method."""
    from ppo_agent import sample_episodes
    return sample_episodes(env, n_episodes, slate_size, np.random.default_rng(seed))


def _episode_mean(rewards):
    return float(np.mean(rewards))


def eval_random(env, eval_set, k, seed):
    profiles, _, slates = eval_set
    rng = np.random.default_rng(seed)
    out = []
    for profile, slate in zip(profiles, slates):
        picks = rng.choice(slate, size=k, replace=False)
        out.extend(env.reward(profile, int(i)) for i in picks)
    return _episode_mean(out)


def eval_ppo(agent, env, eval_set, greedy=True):
    import torch
    profiles, users, slates = eval_set
    k = agent.config.k
    user = torch.from_numpy(users)
    cand = torch.from_numpy(np.asarray(env.item_embeddings)[slates])
    mask = torch.ones(slates.shape, dtype=torch.bool)
    out = []
    for _ in range(k):
        a, _, _ = agent.act(user, cand, mask, greedy=greedy)
        for i, profile in enumerate(profiles):
            out.append(env.reward(profile, int(slates[i, int(a[i])])))
        mask = mask.scatter(1, a.unsqueeze(1), False)
    return _episode_mean(out)


def run_online_mlp(env, embeddings, n_episodes, k, slate_size, eval_set, seed, epsilon=0.1, diversity_penalty=0.2):
    """Train the old online MLP (one gradient step per feedback event) on simulated feedback, then evaluate greedily."""
    import torch

    from recommender_logic import select_indices
    from rl_recommender import RecommendationAgent

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed + 1)
    table = torch.from_numpy(np.asarray(embeddings))
    agent = RecommendationAgent(table.shape[1], table, epsilon=epsilon, diversity_penalty=diversity_penalty,
                                eval_interval=0)

    def play(profile, user_emb, slate, learn):
        history, rewards = [], []
        scores = agent.score_candidates(user_emb).numpy()[slate]
        for _ in range(k):
            local = select_indices(scores, history, top_k=1, epsilon=epsilon if learn else 0.0,
                                   diversity_penalty=diversity_penalty, rng=py_rng)[0]
            history.append(local)
            reward = env.reward(profile, int(slate[local]))
            rewards.append(reward)
            if learn:
                agent.train(user_emb, int(slate[local]), reward)
                scores = agent.score_candidates(user_emb).numpy()[slate]
        return rewards

    for _ in range(n_episodes):
        profile = env.sample_profile(rng)
        slate = rng.choice(env.n_items, size=slate_size, replace=False)
        play(profile, torch.from_numpy(env.user_embedding(profile)), slate, learn=True)

    profiles, users, slates = eval_set
    out = []
    for profile, user_emb, slate in zip(profiles, users, slates):
        out.extend(play(profile, torch.from_numpy(user_emb), slate, learn=False))
    return _episode_mean(out), agent


def plot_curve(rewards, baselines, path):
    """Write the per-update reward curve as SVG with the baselines as dashed lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(rewards) + 1), rewards, label="PPO, training rollouts")
    for label, value in baselines.items():
        ax.axhline(value, linestyle="--", linewidth=1, color="grey" if "random" in label else "tab:orange", label=label)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Mean reward per recommendation")
    ax.set_title("Simulated feedback: PPO training curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def cpu_name():
    return platform.processor() or platform.machine()


def run_experiment(csv_path, output, updates=150, config=None, seed=0, sim_threshold=0.3, eval_episodes=500,
                   cache_dir=DEFAULT_CACHE_DIR, metrics_path=DEFAULT_METRICS_PATH, curve_path=DEFAULT_CURVE_PATH,
                   baseline_output=None, command=None, model=None, log=print):
    """Train, evaluate, save the checkpoint, and write metrics and curve. Returns the metrics dict."""
    import torch

    from ppo_agent import PPOAgent, PPOConfig, collect_rollouts

    config = config or PPOConfig()
    start = time.time()
    df, embeddings = load_catalogue(csv_path, cache_dir, model=model)
    env = build_env(df, embeddings, sim_threshold, seed)
    eval_set = make_eval_set(env, eval_episodes, config.slate_size, seed + 10_000)

    agent = PPOAgent(embeddings.shape[1], config, seed=seed)
    rng = np.random.default_rng(seed)
    per_update = []
    for step in range(1, updates + 1):
        batch, info = collect_rollouts(agent, env, rng)
        stats = agent.update(batch)
        agent.rewards_history.append(info["mean_reward"])
        agent.loss_history.append(stats["total_loss"])
        per_update.append({**info, **stats})
        if step == 1 or step % 10 == 0 or step == updates:
            log(f"update {step}/{updates}: reward {info['mean_reward']:.3f} policy_loss {stats['policy_loss']:.4f} "
                f"value_loss {stats['value_loss']:.4f} entropy {stats['entropy']:.3f}")
    ppo_seconds = time.time() - start

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    agent.save(output)

    ppo_greedy = eval_ppo(agent, env, eval_set, greedy=True)
    ppo_sampled = eval_ppo(agent, env, eval_set, greedy=False)
    random_reward = eval_random(env, eval_set, config.k, seed + 20_000)
    n_episodes = updates * config.episodes_per_update
    mlp_reward, mlp_agent = run_online_mlp(env, embeddings, n_episodes, config.k, config.slate_size, eval_set, seed)
    if baseline_output:
        os.makedirs(os.path.dirname(baseline_output) or ".", exist_ok=True)
        mlp_agent.save_training_data(baseline_output)
    total_seconds = time.time() - start

    metrics = {
        "label": "simulated feedback",
        "command": command,
        "seed": seed,
        "data_source": os.path.basename(csv_path),
        "rows": int(len(df)),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": int(embeddings.shape[1]),
        "hardware": f"CPU ({cpu_name()}), torch {torch.__version__}",
        "wall_time_seconds_ppo_training": round(ppo_seconds, 1),
        "wall_time_seconds_total": round(total_seconds, 1),
        "simulated_user": {"sim_threshold": sim_threshold, "genre_pool_size": len(env.genre_pool),
                           "eval_episodes": eval_episodes},
        "ppo_config": config.__dict__,
        "updates": updates,
        "episodes_trained": n_episodes,
        "final_mean_reward": {
            "unit": "mean raw reward per recommendation on held-out simulated episodes",
            "ppo_greedy": ppo_greedy,
            "ppo_sampled": ppo_sampled,
            "random_policy": random_reward,
            "online_mlp_greedy": mlp_reward,
        },
        "training_last_10_updates_mean_reward": float(np.mean([u["mean_reward"] for u in per_update[-10:]])),
        "mean_reward_per_update": [u["mean_reward"] for u in per_update],
        "mean_episode_return_per_update": [u["mean_episode_return"] for u in per_update],
        "policy_loss_per_update": [u["policy_loss"] for u in per_update],
        "value_loss_per_update": [u["value_loss"] for u in per_update],
        "entropy_per_update": [u["entropy"] for u in per_update],
        "checkpoint": output,
    }
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plot_curve(metrics["mean_reward_per_update"],
               {"random policy": random_reward, "online MLP (greedy, held-out)": mlp_reward}, curve_path)
    return metrics
