import time

import pytest

from nicole_rag import ChaoticRetriever


@pytest.fixture
def retriever():
    return ChaoticRetriever()


def test_adaptive_chaos_adjusts_per_feedback(retriever):
    base_level = retriever.chaos_factor

    retriever.adapt_chaos_from_feedback("creative_user", feedback_score=0.95)
    creative_level = retriever.get_user_chaos_level("creative_user")

    retriever.adapt_chaos_from_feedback("precise_user", feedback_score=0.1)
    precise_level = retriever.get_user_chaos_level("precise_user")

    assert creative_level > base_level
    assert precise_level < base_level
    assert 0.05 <= precise_level <= base_level
    assert base_level <= creative_level <= 0.3


def test_temporal_weighting_decay_is_monotonic(retriever):
    query = "resonance field"
    content = "resonance field dynamics"
    now = time.time()

    fresh = retriever._calculate_relevance(query, content, timestamp=now)
    aged_30 = retriever._calculate_relevance(query, content, timestamp=now - 30 * 86400)
    aged_60 = retriever._calculate_relevance(query, content, timestamp=now - 60 * 86400)

    assert fresh > aged_30 > aged_60

    # Ensure temporal weighting modifies the base relevance
    base_relevance = retriever._calculate_relevance(query, content, timestamp=None)
    assert fresh != pytest.approx(base_relevance)


def test_relevance_zero_when_no_overlap(retriever):
    now = time.time()
    relevance = retriever._calculate_relevance(
        "quantum mind", "classical mechanics", timestamp=now
    )
    assert relevance == pytest.approx(0.3, rel=1e-6)
