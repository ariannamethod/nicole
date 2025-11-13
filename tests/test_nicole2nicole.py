import random

import pytest

from nicole2nicole import LearningPattern, Nicole2NicoleCore


@pytest.fixture
def n2n_core(tmp_path):
    knowledge_file = tmp_path / "knowledge.json"
    return Nicole2NicoleCore(memory_db=":memory:", knowledge_file=str(knowledge_file))


def test_calculate_success_score_matches_expected(n2n_core):
    metrics = {
        "entropy": 0.6,
        "perplexity": 1.2,
        "resonance": 0.7,
        "coherence": 0.8,
        "engagement": 0.5,
    }
    score = n2n_core._calculate_success_score(metrics)

    expected = (
        metrics["entropy"] * 0.2
        + (1.0 / max(0.1, metrics["perplexity"])) * 0.3
        + metrics["resonance"] * 0.3
        + metrics["coherence"] * 0.1
        + metrics["engagement"] * 0.1
    )
    expected = min(1.0, max(0.0, expected))

    assert score == pytest.approx(expected)


def test_learn_from_patterns_builds_preferences(n2n_core):
    patterns = []
    for i in range(6):
        metrics = {
            "entropy": 0.5 + 0.05 * i,
            "perplexity": 1.0 + 0.1 * i,
            "resonance": 0.4 + 0.05 * i,
            "coherence": 0.6,
            "engagement": 0.7,
        }
        architecture = {"learning_rate": 0.01 + 0.005 * i, "temperature": 0.7}
        patterns.append(
            LearningPattern(
                input_pattern="greeting",
                output_pattern="detailed_response",
                metrics_context=metrics,
                architecture_context=architecture,
                success_score=0.5 + 0.05 * i,
            )
        )

    n2n_core.learn_from_patterns(patterns)

    key = "greeting::detailed_response"
    assert key in n2n_core.learning_patterns
    stored = n2n_core.learning_patterns[key]
    assert stored["frequency"] == len(patterns)
    assert pytest.approx(stored["avg_success"]) == sum(p.success_score for p in patterns) / len(patterns)

    preferences = n2n_core.architecture_preferences
    assert "learning_rate" in preferences
    lr_pref = preferences["learning_rate"]
    assert lr_pref["min"] <= lr_pref["avg"] <= lr_pref["max"]
    assert lr_pref["samples"] >= 1


def test_suggest_response_strategy_prefers_best_pattern(n2n_core):
    patterns = [
        LearningPattern(
            input_pattern="greeting",
            output_pattern="brief_response",
            metrics_context={},
            architecture_context={"learning_rate": 0.01},
            success_score=0.3,
        ),
        LearningPattern(
            input_pattern="greeting",
            output_pattern="brief_response",
            metrics_context={},
            architecture_context={"learning_rate": 0.011},
            success_score=0.35,
        ),
        LearningPattern(
            input_pattern="greeting",
            output_pattern="detailed_response",
            metrics_context={},
            architecture_context={"learning_rate": 0.02},
            success_score=0.75,
        ),
        LearningPattern(
            input_pattern="greeting",
            output_pattern="detailed_response",
            metrics_context={},
            architecture_context={"learning_rate": 0.021},
            success_score=0.82,
        ),
    ]

    n2n_core.learn_from_patterns(patterns)

    recommendation = n2n_core.suggest_response_strategy("hello Nicole", {})
    assert recommendation == "detailed_response"


def test_suggest_architecture_improvements_respects_preferences(n2n_core):
    patterns = []
    for i in range(5):
        metrics = {
            "entropy": 0.5,
            "perplexity": 1.0 + 0.05 * i,
            "resonance": 0.6 + 0.05 * i,
            "coherence": 0.7,
            "engagement": 0.6,
        }
        architecture = {
            "learning_rate": 0.01 + 0.002 * i,
            "temperature": 0.6 + 0.01 * i,
        }
        patterns.append(
            LearningPattern(
                input_pattern="information_request",
                output_pattern="detailed_response",
                metrics_context=metrics,
                architecture_context=architecture,
                success_score=0.4 + 0.05 * i,
            )
        )

    random.seed(0)
    n2n_core.learn_from_patterns(patterns)

    current_arch = {"learning_rate": 0.005, "temperature": 0.65}
    improved = n2n_core.suggest_architecture_improvements(current_arch, "context")

    assert improved["learning_rate"] >= current_arch["learning_rate"]
    assert abs(improved["temperature"] - current_arch["temperature"]) <= 0.65 * 0.05
