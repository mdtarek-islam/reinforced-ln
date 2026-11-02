"""Plot reward progression and loss reduction from a saved training checkpoint.

PyTorch and matplotlib load inside the functions that need them.
``summarise_history`` needs neither.
"""
import os

DEFAULT_STATS_PATH = os.path.join(os.environ.get("REINFORCEDLN_MODELS_DIR") or "models", "agent_stats.pth")


def load_training_data(path=DEFAULT_STATS_PATH):
    """Loads saved training data from file.

    ``weights_only=True`` restricts unpickling to tensors and plain containers.
    Only load checkpoints from a source you trust.
    """
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint


def summarise_history(rewards_history, loss_history):
    """Return simple numbers that describe a training run.

    Keys: steps, positive, negative, mean_reward, first_loss, last_loss, min_loss.
    Loss keys are None when the loss history is empty.
    """
    rewards = list(rewards_history)
    losses = list(loss_history)
    steps = len(rewards)
    return {
        "steps": steps,
        "positive": sum(1 for r in rewards if r > 0),
        "negative": sum(1 for r in rewards if r < 0),
        "mean_reward": (sum(rewards) / steps) if steps else None,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
    }


def _finish(plt, save_path):
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_rewards(rewards_history, save_path=None):
    """Plots reward progression. Saves to ``save_path`` when given, else shows it."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history, label="Reward Trend", color="blue")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.title("Reward Progression Over Training")
    plt.legend()
    _finish(plt, save_path)


def plot_loss(loss_history, save_path=None):
    """Plots loss reduction over time. Saves to ``save_path`` when given, else shows it."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Loss Trend", color="red")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Over Time")
    plt.legend()
    _finish(plt, save_path)


def show_model_architecture(path=DEFAULT_STATS_PATH, emb_dim=384):
    """Displays the RL Neural Network architecture.

    ``emb_dim`` defaults to 384, the SBERT all-MiniLM-L6-v2 embedding size.
    """
    from torchsummary import summary
    from rl_recommender import RLRecommender  # Load model class

    model = RLRecommender(emb_dim=emb_dim)
    checkpoint = load_training_data(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # torchsummary defaults to CUDA; force CPU so it runs on any machine
    summary(model, (1, model.fc1.in_features), device="cpu")


if __name__ == "__main__":
    # Load saved data
    data = load_training_data()

    # Plot results
    plot_rewards(data['rewards_history'])
    plot_loss(data['loss_history'])
    show_model_architecture()
