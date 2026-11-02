from data_loader import load_and_process_data, sbert_model
from rl_recommender import RecommendationAgent
from ui_feedback import run_feedback_ui
import torch
import numpy as np

if __name__ == "__main__":
    # Load and process the reduced dataset
    df, embeddings = load_and_process_data()
    # Convert embeddings to a torch tensor
    embeddings_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
    emb_dim = embeddings_tensor.shape[1]
    num_items = len(df)
    
    # Initialize the RL agent with the dataset embeddings
    agent = RecommendationAgent(emb_dim, embeddings_tensor)
    
    # Run the Gradio UI, passing the SBERT model for user input encoding
    run_feedback_ui(df, embeddings_tensor, agent, sbert_model)
