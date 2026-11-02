"""Neural scoring network and feedback-driven recommendation agent.

This module needs PyTorch. The epsilon-greedy policy and the diversity penalty
live in ``recommender_logic.py`` so they can run without PyTorch.
"""
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from recommender_logic import select_indices

# Weights go to a gitignored models/ folder. REINFORCEDLN_MODELS_DIR overrides it.
DEFAULT_STATS_PATH = os.path.join(os.environ.get("REINFORCEDLN_MODELS_DIR") or "models", "agent_stats.pth")


class RLRecommender(nn.Module):
    """
    Neural Network that scores a candidate given a concatenated vector of the
    user embedding and the candidate's embedding.
    Input dimension is 2 * embedding_dimension; output is a single score.
    """
    def __init__(self, emb_dim):
        super(RLRecommender, self).__init__()
        input_dim = 2 * emb_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output shape: (batch_size, 1)

class RecommendationAgent:
    """
    This agent uses the RLRecommender network to score each candidate light novel.
    Given a user embedding, it forms a batch by concatenating the user embedding with each candidate's embedding.
    An epsilon-greedy strategy is applied for exploration, and a diversity penalty is applied to recently recommended items.

    ``stats_path`` sets where ``save_training_data`` writes the checkpoint.
    The default file name is ``agent_stats.pth`` inside the models folder.
    """
    def __init__(self, emb_dim, dataset_embeddings, epsilon=0.1, diversity_penalty=0.2, eval_interval=30,
                 stats_path=DEFAULT_STATS_PATH):
        self.emb_dim = emb_dim
        self.dataset_embeddings = dataset_embeddings
        self.num_items = dataset_embeddings.shape[0]
        self.model = RLRecommender(emb_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.diversity_penalty = diversity_penalty
        self.history = []
        self.stats_path = stats_path

        # Logs for evaluation
        self.reward_log = []
        self.loss_log = []
        self.eval_interval = eval_interval  # Save a checkpoint every `eval_interval` feedback interactions
        self.train_step = 0  # Track training steps

        self.training_steps = 0
        self.rewards_history = []  # Store rewards
        self.loss_history = []  # Store loss values

    def score_candidates(self, user_embedding):
        """Return the network score of every candidate as a 1-D tensor."""
        # Expand user_embedding to shape (num_items, emb_dim)
        user_batch = user_embedding.unsqueeze(0).repeat(self.num_items, 1)
        # Concatenate with candidate embeddings: shape (num_items, 2*emb_dim)
        input_tensor = torch.cat([user_batch, self.dataset_embeddings], dim=1)
        with torch.no_grad():
            return self.model(input_tensor).reshape(-1)  # shape: (num_items,)

    def select_actions(self, user_embedding, top_k=3):
        """
        Given a user embedding (tensor of shape (emb_dim,)),
        builds inputs for all candidates (concatenates user embedding with each candidate's embedding),
        then returns the indices of the top_k candidates with the highest scores.
        Epsilon-greedy exploration returns random candidates with probability epsilon.
        A diversity penalty lowers the scores of the last 5 recommended items.
        """
        scores = self.score_candidates(user_embedding).numpy()
        return select_indices(
            scores,
            self.history,
            top_k=top_k,
            epsilon=self.epsilon,
            diversity_penalty=self.diversity_penalty,
            rng=random,
        )

    def train(self, user_embedding, candidate_idx, reward):
        """Run one gradient step that moves the candidate's score towards ``reward``.

        The loss is the mean squared error between the predicted score and the
        reward (+1 or -1). Every ``eval_interval`` steps the agent saves a checkpoint.
        Returns the loss value.
        """
        candidate_embedding = self.dataset_embeddings[candidate_idx].unsqueeze(0)
        user_batch = user_embedding.unsqueeze(0)
        input_tensor = torch.cat([user_batch, candidate_embedding], dim=1)
        predicted_score = self.model(input_tensor).squeeze()
        target = torch.tensor(reward, dtype=torch.float32)
        loss = self.criterion(predicted_score, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store training data
        self.training_steps += 1
        self.rewards_history.append(reward)
        self.loss_history.append(loss.item())

        # Save progress every `eval_interval` steps (30 by default)
        if self.eval_interval and self.training_steps % self.eval_interval == 0:
            self.save_training_data()
        return loss.item()

    def save_training_data(self, path=None):
        """Saves model and training data for visualisation."""
        path = path or self.stats_path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rewards_history': self.rewards_history,
            'loss_history': self.loss_history,
            'training_steps': self.training_steps
        }, path)
        print(f"Training data saved at step {self.training_steps}")

    def load_model_weights(self, path):
        """Load network weights from a checkpoint written by ``save_training_data``.

        ``weights_only=True`` stops the loader from running arbitrary pickled code.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
