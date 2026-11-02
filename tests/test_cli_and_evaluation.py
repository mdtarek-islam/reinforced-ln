import pytest

import cli
from evaluation_visuals import summarise_history


@pytest.mark.parametrize("answer, expected", [("g", "Good"), (" YES ", "Good"), ("b", "Bad"), ("no", "Bad"), ("q", None)])
def test_parse_feedback(answer, expected):
    assert cli.parse_feedback(answer) == expected


def test_parse_feedback_rejects_unknown():
    with pytest.raises(ValueError):
        cli.parse_feedback("maybe")


def test_ask_feedback_retries_until_valid(capsys):
    answers = iter(["?", "bad"])
    assert cli.ask_feedback("> ", input_fn=lambda _: next(answers)) == "Bad"
    assert "Unrecognised" in capsys.readouterr().out


def test_parser_subcommands():
    parser = cli.build_parser()
    train = parser.parse_args(["train", "--rounds", "2", "--seed", "1"])
    assert (train.command, train.rounds, train.seed, train.ui) == ("train", 2, 1, False)
    assert train.algo == "ppo"
    assert parser.parse_args(["train", "--algo", "online"]).algo == "online"
    rec = parser.parse_args(["recommend", "a hero story", "--top-k", "5"])
    assert (rec.synopsis, rec.top_k, rec.epsilon) == ("a hero story", 5, 0.0)
    ev = parser.parse_args(["evaluate", "--stats", "x.pth", "--no-plots"])
    assert (ev.stats, ev.no_plots) == ("x.pth", True)
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_summarise_history():
    s = summarise_history([1, -1, 1, 1], [0.9, 0.7, 0.5, 0.6])
    assert s == {"steps": 4, "positive": 3, "negative": 1, "mean_reward": 0.5,
                 "first_loss": 0.9, "last_loss": 0.6, "min_loss": 0.5}


def test_summarise_empty_history():
    s = summarise_history([], [])
    assert s["steps"] == 0 and s["mean_reward"] is None and s["last_loss"] is None


def test_plots_write_files(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import evaluation_visuals as ev
    ev.plot_rewards([1, -1, 1], save_path=str(tmp_path / "r.png"))
    ev.plot_loss([0.9, 0.5], save_path=str(tmp_path / "l.png"))
    assert (tmp_path / "r.png").exists() and (tmp_path / "l.png").exists()


def test_train_defaults_to_ppo_with_documented_hyperparameters():
    args = cli.build_parser().parse_args(["train"])
    assert (args.algo, args.clip_eps, args.gamma, args.lam, args.k) == ("ppo", 0.2, 0.99, 0.95, 5)
    assert args.output is None  # resolved to the models folder at run time
