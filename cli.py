"""Command-line entry point for ReinforcedLN.

Subcommands:
  train      train the agent. Default: offline PPO against a simulated user.
             With --algo online: collect Good or Bad feedback in the terminal (or the Gradio UI)
  recommend  print the top recommendations for one synopsis
  evaluate   summarise and plot the reward and loss history of a checkpoint

``main.py`` keeps the original flow: it launches the Gradio feedback UI.
Heavy libraries (PyTorch, SBERT, Gradio, matplotlib) load only when a
subcommand needs them.
"""
import argparse
import os
import sys

from recommender_logic import NEGATIVE_FEEDBACK, POSITIVE_FEEDBACK

DEFAULT_DATA_PATH = "reduced_light_novels.csv"
DEFAULT_SYNOPSES_PATH = "synopses.txt"
MODELS_DIR_ENV = "REINFORCEDLN_MODELS_DIR"
# Weights go to a gitignored models/ folder. Set REINFORCEDLN_MODELS_DIR or pass --output to move them.
DEFAULT_MODELS_DIR = os.environ.get(MODELS_DIR_ENV) or "models"
DEFAULT_TRAIN_OUTPUT = os.path.join(DEFAULT_MODELS_DIR, "agent_stats.pth")
DEFAULT_PPO_OUTPUT = os.path.join(DEFAULT_MODELS_DIR, "ppo_agent.pt")


def parse_feedback(answer):
    """Map a terminal answer to "Good", "Bad" or None (quit).

    Accepts g/good/y/yes, b/bad/n/no and q/quit, in any case.
    Raises ValueError for anything else.
    """
    value = answer.strip().lower()
    if value in ("g", "good", "y", "yes"):
        return POSITIVE_FEEDBACK
    if value in ("b", "bad", "n", "no"):
        return NEGATIVE_FEEDBACK
    if value in ("q", "quit"):
        return None
    raise ValueError(f"Unrecognised answer: {answer!r}")


def ask_feedback(prompt, input_fn=input):
    """Prompt until the answer parses. Return "Good", "Bad" or None."""
    while True:
        try:
            return parse_feedback(input_fn(prompt))
        except ValueError as exc:
            print(exc)


def build_parser():
    """Return the argparse parser for the three subcommands."""
    parser = argparse.ArgumentParser(prog="reinforcedln", description="Light novel recommender with feedback-driven training.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the agent: PPO on simulated feedback, or online Good or Bad feedback.")
    train.add_argument("--algo", choices=("ppo", "online"), default="ppo",
                       help="ppo: offline PPO against a simulated user (default). online: the original one-step MLP loop.")
    train.add_argument("--data", default=DEFAULT_DATA_PATH, help="Dataset CSV.")
    train.add_argument("--synopses", default=DEFAULT_SYNOPSES_PATH, help="Text file of test synopses.")
    train.add_argument("--rounds", type=int, default=5, help="Recommendations rated per synopsis.")
    train.add_argument("--epsilon", type=float, default=0.1, help="Exploration probability.")
    train.add_argument("--checkpoint", help="Start from the weights in this checkpoint.")
    train.add_argument("--output", help=f"Checkpoint path. Default: {DEFAULT_PPO_OUTPUT} (ppo) or {DEFAULT_TRAIN_OUTPUT} (online).")
    train.add_argument("--seed", type=int, help="Random seed (ppo default: 0).")
    train.add_argument("--ui", action="store_true", help="With --algo online: launch the Gradio UI in place of the terminal loop.")
    ppo = train.add_argument_group("ppo options (simulated feedback)")
    ppo.add_argument("--updates", type=int, default=150, help="PPO updates (rollout batches).")
    ppo.add_argument("--episodes-per-update", type=int, default=64)
    ppo.add_argument("--k", type=int, default=5, help="Recommendations per episode.")
    ppo.add_argument("--slate-size", type=int, default=30, help="Candidate novels offered per step.")
    ppo.add_argument("--epochs", type=int, default=4, help="Passes over each batch.")
    ppo.add_argument("--minibatch-size", type=int, default=64)
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--lam", type=float, default=0.95, help="GAE lambda.")
    ppo.add_argument("--clip-eps", type=float, default=0.2)
    ppo.add_argument("--entropy-coef", type=float, default=0.01)
    ppo.add_argument("--diversity-penalty", type=float, default=0.2, help="Reward shaping for repeated genre strings.")
    ppo.add_argument("--sim-threshold", type=float, default=0.3, help="Synopsis-similarity gate of the simulated user.")
    ppo.add_argument("--eval-episodes", type=int, default=500)
    ppo.add_argument("--cache-dir", default="embeddings_cache", help="Folder for cached SBERT embeddings.")
    ppo.add_argument("--metrics", default=os.path.join("docs", "local", "repro-metrics.json"))
    ppo.add_argument("--curve", default=os.path.join("docs", "local", "ppo-training-curve.svg"))
    ppo.add_argument("--baseline-output", help="Also save the online MLP baseline checkpoint here.")

    rec = sub.add_parser("recommend", help="Print recommendations for a synopsis.")
    rec.add_argument("synopsis", help="Free-text story description.")
    rec.add_argument("--data", default=DEFAULT_DATA_PATH, help="Dataset CSV.")
    rec.add_argument("--top-k", type=int, default=3, help="Number of recommendations.")
    rec.add_argument("--epsilon", type=float, default=0.0, help="Exploration probability (0 is greedy).")
    rec.add_argument("--checkpoint", help="Load weights from this trusted checkpoint.")
    rec.add_argument("--seed", type=int, help="Random seed.")

    ev = sub.add_parser("evaluate", help="Summarise and plot a training checkpoint.")
    ev.add_argument("--stats", default=DEFAULT_TRAIN_OUTPUT, help="Trusted checkpoint written by training.")
    ev.add_argument("--save-dir", help="Write PNG plots here in place of showing them.")
    ev.add_argument("--no-plots", action="store_true", help="Print the summary only.")
    ev.add_argument("--architecture", action="store_true", help="Print the network summary (needs torchsummary).")
    return parser


def _seed(seed):
    if seed is None:
        return
    import random

    import torch
    random.seed(seed)
    torch.manual_seed(seed)


def _build_agent(data_path, epsilon, checkpoint=None, stats_path=None):
    """Load data, embed it with SBERT and return (df, agent, sbert_model)."""
    import numpy as np
    import torch

    from data_loader import get_sbert_model, load_and_process_data
    from rl_recommender import DEFAULT_STATS_PATH, RecommendationAgent

    sbert_model = get_sbert_model()
    df, embeddings = load_and_process_data(data_path, model=sbert_model)
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    agent = RecommendationAgent(
        embeddings_tensor.shape[1], embeddings_tensor, epsilon=epsilon,
        stats_path=stats_path or DEFAULT_STATS_PATH,
    )
    if checkpoint:
        agent.load_model_weights(checkpoint)
    return df, agent, sbert_model


def _embed(sbert_model, text):
    import torch
    return torch.tensor(sbert_model.encode([text]), dtype=torch.float32).squeeze(0)


def cmd_train_ppo(args, argv=None):
    """Offline PPO against the simulated user. Results are labelled "simulated feedback"."""
    from ppo_agent import PPOConfig
    from ppo_training import run_experiment

    config = PPOConfig(
        k=args.k, slate_size=args.slate_size, episodes_per_update=args.episodes_per_update, epochs=args.epochs,
        minibatch_size=args.minibatch_size, lr=args.lr, gamma=args.gamma, lam=args.lam, clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef, diversity_penalty=args.diversity_penalty,
    )
    command = "python cli.py " + " ".join(argv if argv is not None else sys.argv[1:])
    metrics = run_experiment(
        args.data, args.output or DEFAULT_PPO_OUTPUT, updates=args.updates, config=config,
        seed=args.seed if args.seed is not None else 0, sim_threshold=args.sim_threshold,
        eval_episodes=args.eval_episodes, cache_dir=args.cache_dir, metrics_path=args.metrics,
        curve_path=args.curve, baseline_output=args.baseline_output, command=command,
    )
    final = metrics["final_mean_reward"]
    print("Simulated feedback, mean reward per recommendation on held-out episodes:")
    for name, value in final.items():
        if name != "unit":
            print(f"  {name}: {value:.3f}")
    return 0


def cmd_train(args, argv=None):
    """Run PPO on simulated feedback (default), or the terminal feedback loop / Gradio UI with ``--algo online``."""
    if args.algo == "ppo":
        return cmd_train_ppo(args, argv)
    args.output = args.output or DEFAULT_TRAIN_OUTPUT
    _seed(args.seed)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df, agent, sbert_model = _build_agent(args.data, args.epsilon, args.checkpoint, args.output)

    if args.ui:
        from ui_feedback import run_feedback_ui
        run_feedback_ui(df, agent.dataset_embeddings, agent, sbert_model)
        return 0

    import training_log
    from data_loader import load_synopses
    from recommender_logic import feedback_to_reward

    for number, synopsis in enumerate(load_synopses(args.synopses), start=1):
        print(f"\nSynopsis {number}: {synopsis}")
        user_embedding = _embed(sbert_model, synopsis)
        training_log.record(synopsis, user_embedding.numpy())
        for _ in range(args.rounds):
            indices = agent.select_actions(user_embedding)
            agent.history.extend(indices)
            rec = df.iloc[indices[0]]
            print(f"\n  {rec['Title']}\n  Genres: {rec['Genres']}")
            feedback = ask_feedback("  Rate it [g]ood, [b]ad or [q]uit: ")
            if feedback is None:
                agent.save_training_data(args.output)
                return 0
            loss = agent.train(user_embedding, indices[0], feedback_to_reward(feedback))
            print(f"  Step {agent.training_steps}: loss {loss:.4f}")

    agent.save_training_data(args.output)
    return 0


def cmd_recommend(args):
    """Print the ranked recommendations for one synopsis."""
    _seed(args.seed)
    df, agent, sbert_model = _build_agent(args.data, args.epsilon, args.checkpoint)
    indices = agent.select_actions(_embed(sbert_model, args.synopsis), top_k=args.top_k)
    for rank, idx in enumerate(indices, start=1):
        rec = df.iloc[idx]
        print(f"{rank}. {rec['Title']} [{rec['Genres']}]")
    return 0


def cmd_evaluate(args):
    """Print a summary of the checkpoint history and plot it."""
    import evaluation_visuals as ev

    data = ev.load_training_data(args.stats)
    summary = ev.summarise_history(data["rewards_history"], data["loss_history"])
    for key, value in summary.items():
        print(f"{key}: {value}")

    if not args.no_plots:
        reward_path = loss_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            reward_path = os.path.join(args.save_dir, "reward_progression.png")
            loss_path = os.path.join(args.save_dir, "loss_reduction.png")
        ev.plot_rewards(data["rewards_history"], save_path=reward_path)
        ev.plot_loss(data["loss_history"], save_path=loss_path)
    if args.architecture:
        ev.show_model_architecture(args.stats)
    return 0


COMMANDS = {"train": cmd_train, "recommend": cmd_recommend, "evaluate": cmd_evaluate}


def main(argv=None):
    """Parse ``argv`` and run the chosen subcommand. Returns an exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return cmd_train(args, argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
