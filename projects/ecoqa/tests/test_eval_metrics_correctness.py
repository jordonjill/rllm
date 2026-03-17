from types import SimpleNamespace

from projects.ecoqa.run_ecoqa import _print_eval_metrics, _trajectory_is_correct as run_is_correct
from projects.ecoqa.run_ecoqa_benchmark import _compute_metrics, _trajectory_is_correct as benchmark_is_correct


def _trajectory(
    question_id: str,
    *,
    reward: float,
    is_correct: bool | None = None,
    correctness_reward: float | None = None,
):
    info = {}
    if is_correct is not None:
        info["is_correct"] = is_correct
    if correctness_reward is not None:
        info["metadata"] = {"correctness_reward": correctness_reward}

    return SimpleNamespace(
        task={
            "question_id": question_id,
            "question_type": "single_table",
            "answer_type": "scalar",
        },
        reward=reward,
        steps=[SimpleNamespace(info=info)],
    )


def test_benchmark_trajectory_is_correct_ignores_positive_shaping_reward():
    traj = _trajectory("q1", reward=0.15, is_correct=False)
    assert benchmark_is_correct(traj) is False


def test_benchmark_metrics_use_is_correct_flag_not_reward_sign():
    trajectories = [
        _trajectory("q1", reward=0.15, is_correct=False),
        _trajectory("q1", reward=1.0, is_correct=True),
        _trajectory("q2", reward=0.12, is_correct=False),
    ]
    metrics = _compute_metrics(trajectories)
    assert metrics["num_trajectories"] == 3
    assert metrics["num_unique_questions"] == 2
    assert metrics["pass_at_1"] == 1 / 3
    assert metrics["pass_at_k"] == 1 / 2
    assert metrics["accuracy_by_type"]["scalar"] == 1 / 3


def test_run_metrics_use_is_correct_flag_not_reward_sign(capsys):
    trajectories = [
        _trajectory("q1", reward=0.2, is_correct=False),
        _trajectory("q1", reward=0.1, is_correct=False),
        _trajectory("q2", reward=0.05, is_correct=True),
    ]

    _print_eval_metrics(trajectories)
    captured = capsys.readouterr().out

    assert "Total unique problems: 2" in captured
    assert "Pass@1: 0.3333333333333333" in captured
    assert "Pass@k: 0.5" in captured


def test_run_is_correct_supports_correctness_reward_metadata_fallback():
    correct = _trajectory("q1", reward=0.0, correctness_reward=1.0)
    incorrect = _trajectory("q2", reward=0.2, correctness_reward=0.0)
    assert run_is_correct(correct) is True
    assert run_is_correct(incorrect) is False
