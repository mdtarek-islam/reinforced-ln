"""Gradio interface that shows recommendations and collects feedback.

Gradio, PyTorch, requests and Pillow load inside the functions that need them,
so the formatting helpers here import without those libraries.
"""
from io import BytesIO

import pandas as pd

import training_log  # import the log module
from recommender_logic import NEGATIVE_FEEDBACK, POSITIVE_FEEDBACK, feedback_to_reward

IMAGE_TIMEOUT_SECONDS = 10


def format_recommendation(rec):
    """Return Markdown text with the title, genres and synopsis of one novel."""
    return (
        f"### {rec['Title']}\n"
        f"**Genres:** {rec['Genres']}\n\n"
        f"**Synopsis:** {rec['Synopsis']}"
    )


def cover_url_from_record(rec):
    """Return the medium cover URL with forward slashes, or None if absent."""
    url = rec.get("Medium_Picture")
    if url is None or pd.isna(url) or not str(url).strip() or url == "N/A":
        return None
    return str(url).replace("\\", "/")


def fetch_cover_image(cover_url):
    """Download a cover image and return it as a PIL image, or None on failure."""
    if cover_url is None:
        return None
    import requests
    from PIL import Image

    try:
        response = requests.get(cover_url, timeout=IMAGE_TIMEOUT_SECONDS)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print("Error loading image from URL:", e)
        return None


def run_feedback_ui(df, embeddings, agent, sbert_model):
    """Creates an interactive UI for recommendations using SBERT and RL."""
    import gradio as gr
    import torch

    def recommend(user_synopsis):
        """Generates recommendations based on the user-entered synopsis."""
        if not user_synopsis or not user_synopsis.strip():
            return "Please enter a synopsis to get recommendations.", None, None, None

        # Embed user input using SBERT
        with torch.no_grad():
            user_embedding = torch.tensor(sbert_model.encode([user_synopsis]), dtype=torch.float32).squeeze(0)

        # Record the synopsis and embedding for evaluation
        training_log.record(user_synopsis, user_embedding.numpy())

        # Use RL agent to select actions based on the embedding
        recommended_idx = agent.select_actions(user_embedding)
        agent.history.extend(recommended_idx)  # update history with recommended indices

        # Show details for the first recommended item
        rec = df.iloc[recommended_idx[0]]
        img = fetch_cover_image(cover_url_from_record(rec))

        # Keep the row position, not the title: titles repeat in the dataset
        return format_recommendation(rec), img, int(recommended_idx[0]), user_embedding

    def update_model(feedback, recommended_idx, user_embedding):
        """Trains the model based on user feedback."""
        if recommended_idx is None or user_embedding is None:
            return "No recommendation to provide feedback on!", None

        reward = feedback_to_reward(feedback)  # Simple +1/-1 reward
        recommended_title = df.iloc[recommended_idx]["Title"]

        # Update the RL agent using the stored user embedding
        agent.train(user_embedding, recommended_idx, reward)
        return f"Feedback received for '{recommended_title}'! Get another recommendation.", None

    # Gradio UI
    with gr.Blocks() as ui:
        gr.Markdown("## Light Novel Recommendation System")
        gr.Markdown("Enter a synopsis to get personalised recommendations!")

        user_input = gr.Textbox(label="Enter a synopsis")
        recommend_button = gr.Button("Recommend")

        with gr.Row():
            with gr.Column(scale=1):
                cover_image = gr.Image(label="Cover Image", interactive=False)
            with gr.Column(scale=2):
                recommendation_text = gr.Markdown()

        # States for the recommended row position and the last user embedding
        recommended_idx_state = gr.State(value=None)
        last_user_embedding_state = gr.State(value=None)

        recommend_button.click(
            recommend,
            inputs=[user_input],
            outputs=[recommendation_text, cover_image, recommended_idx_state, last_user_embedding_state]
        )

        feedback_message = gr.Textbox(label="Feedback Status", interactive=False)

        with gr.Row():
            feedback_good = gr.Button(POSITIVE_FEEDBACK)
            feedback_bad = gr.Button(NEGATIVE_FEEDBACK)

        feedback_good.click(
            lambda: POSITIVE_FEEDBACK,
            outputs=[feedback_message]
        ).then(
            update_model,
            inputs=[feedback_message, recommended_idx_state, last_user_embedding_state],
            outputs=[feedback_message, last_user_embedding_state]
        )

        feedback_bad.click(
            lambda: NEGATIVE_FEEDBACK,
            outputs=[feedback_message]
        ).then(
            update_model,
            inputs=[feedback_message, recommended_idx_state, last_user_embedding_state],
            outputs=[feedback_message, last_user_embedding_state]
        )

        ui.launch()
