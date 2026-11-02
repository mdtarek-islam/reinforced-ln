# Specification trace

Sources: `reinforcedln_prototype.pdf` (10 pages) and `reinforcedln_report.pdf` (5 pages).
Page numbers are the printed page numbers in each PDF.

## Requirements

| ID | Requirement (source) | Code | Test |
|---|---|---|---|
| R1 | Load the reduced CSV and join 'Genres' and 'Synopsis' into one text, with missing data handled (prototype p3) | `data_loader.build_text_representation`, `data_loader.load_dataset` | `tests/test_data_loader.py` |
| R2 | Embed each novel with SBERT "all-MiniLM-L6-v2" (prototype p3) | `data_loader.get_sbert_model`, `data_loader.load_and_process_data` | `test_load_and_process_data_uses_given_model` (fake encoder) |
| R3 | Score the concatenated user and candidate embeddings with layers 2d to 128 to 64 to 1, all trainable (prototype p4) | `rl_recommender.RLRecommender` | `test_network_layers_match_spec` (needs torch) |
| R4 | Explore with an epsilon-greedy policy (prototype p4, report p3) | `recommender_logic.select_indices` | `test_exploration_returns_distinct_random_indices`, `test_epsilon_controls_exploration_rate` |
| R5 | Penalise recently recommended items for diversity (prototype p4, report p3) | `recommender_logic.apply_diversity_penalty` | `test_penalty_hits_only_the_last_window`, `test_greedy_selection_respects_diversity_penalty` |
| R6 | Reward +1 for positive and -1 for negative feedback (prototype p5) | `recommender_logic.feedback_to_reward` | `test_feedback_to_reward` |
| R7 | Update the model from each feedback event (prototype p5) | `RecommendationAgent.train` | `test_positive_feedback_raises_score_and_lowers_loss` (needs torch) |
| R8 | Rank candidates and return the top three (report p4) | `recommender_logic.rank_top_k`, `RecommendationAgent.select_actions` | `test_rank_top_k_orders_best_first_and_breaks_ties_by_index`, `test_greedy_selection_matches_score_ranking` |
| R9 | Gradio UI: enter a synopsis, show title, synopsis and cover of the top item, collect Good or Bad (prototype p5) | `ui_feedback.run_feedback_ui`, `format_recommendation`, `cover_url_from_record` | `tests/test_ui_and_extractor.py` (helpers only) |
| R10 | Plot reward progression and loss reduction (prototype p8 to 9) | `evaluation_visuals.plot_rewards`, `plot_loss`, `summarise_history` | `test_summarise_history`, `test_plots_write_files` (needs matplotlib) |
| R11 | `main.py` loads data, builds the agent and launches the UI (prototype p8) | `main.py` (unchanged) | Not tested: needs SBERT and Gradio |
| R12 | Fetch light novels from the MyAnimeList API (report p4) | `dataset_extraction/extractor.py` | `test_node_to_row_handles_null_fields` |
| R13 | Reduce to about 2,000 entries, 80:20 with and without synopses, balanced genres (report p4) | `dataset_extraction/reducer.reduce_dataset` | `tests/test_reducer.py` |
| R14 | Log user synopses and embeddings for evaluation | `training_log.record`, `training_log.reset` | `tests/test_training_log.py` |
| R15 | Six synopses, five recommendations each, 30 feedback trials (prototype p7) | `cli.py train --rounds 5` over `synopses.txt` | `test_parser_subcommands` |
| R16 | Optimise the policy with Proximal Policy Optimisation (report p4): actor-critic, GAE, clipped surrogate, value loss, entropy bonus, minibatch epochs, diversity penalty as reward shaping | `ppo_agent.ActorCritic`, `compute_gae`, `clipped_surrogate_loss`, `PPOAgent.update`, `collect_rollouts`; `cli.py train` (default `--algo ppo`); offline run in `ppo_training.run_experiment` | `tests/test_ppo_agent.py` (GAE example, clip gradient direction, probabilities sum to 1, one update improves reward, checkpoint with `weights_only=True`), `tests/test_ppo_training.py` |
| R17 | Simulated user for offline training: hidden genre profile, +1 for a shared genre above a synopsis-similarity threshold, else -1. Results are labelled "simulated feedback" | `simulated_user.SimulatedUser` | `tests/test_simulated_user.py` |

## Results stated in the PDFs

1. Prototype p7: "A series of tests were conducted using six different synopses of varying lengths". Each synopsis received five recommendations, "leading to 30 feedback trials".
2. Prototype p9: "Starting from an initial loss value of approximately 0.95, the loss steadily decreases to around 0.43 after 30 feedback trials."
3. Report p4: the source dataset holds "over 10,000 light novels". The MVP set is "streamlined to around 2,000 entries, maintaining an 80:20 ratio".

The PDFs state no accuracy, precision or recall figures. This repo does not reproduce the loss figures above.

## Divergences between the PDFs and the code

1. Report p4 names Proximal Policy Optimisation. PPO is now implemented (R16) and is the default for `cli.py train`. It trains offline against a simulated user. The original online MLP path (mean squared error, one step per feedback event) stays behind `--algo online`, and `main.py` and the Gradio UI still use it.
2. Report p4 says the UI shows the top three. The UI computes three and displays the first, as prototype p5 describes.
3. `synopses.txt` holds 10 synopses. Prototype p7 reports trials on six.
4. The shipped `reduced_light_novels.csv` has 2,000 rows but 398 repeated IDs. The fixed reducer removes repeats. The CSV stays as shipped.
5. `requirements.txt` lists `gymnasium` and `stable-baselines3`. No module imports them. They stay in the file as originally listed.

## Bugs fixed

1. `reducer.py` drew a multi-genre novel once per genre, so the output held duplicates. It now keeps one copy.
2. `reducer.py` iterated a set of genres, so output order depended on string hash seeds. Genres are now sorted.
3. `reducer.py` ran at import. It now runs from `main()`.
4. `extractor.py` crashed when the API returned a null synopsis or picture. It also held a hard-coded client ID. It now reads `MAL_CLIENT_ID`.
5. `ui_feedback.py` found the rated novel by title. Titles repeat, so feedback could hit the wrong row. It now stores the row position.
6. `ui_feedback.py` crashed on a missing cover URL and had no request timeout. Both are fixed.
7. `evaluation_visuals.py` called torchsummary with its CUDA default and printed `None`. It now runs on CPU. `torch.load` uses `weights_only=True`.
8. `RecommendationAgent.select_actions` failed for a single candidate after `squeeze()`. It now uses `reshape(-1)`.
9. `data_loader.py` loaded SBERT at import. It now loads on first use, and `from data_loader import sbert_model` still works.

## Verification

Run: `".venv/Scripts/python.exe" -m pytest "reinforced-ln" -q`

Result on 2026-11-02: see the final test run in the hand-off. No test is skipped now that torch and matplotlib are installed.

PPO training measurements use simulated feedback only. They stay in the local `docs/local/repro-metrics.json` and are not quoted here.
