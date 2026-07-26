from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from clear.models.claim import BeliefState, LearnerStateClaim
from clear.types import (
    AdmissionStatus,
    ConstructType,
    InferenceLevel,
    ScopeGranularity,
    SourceChannel,
)


def make_claim(**overrides):
    data = {
        "claim_id": "claim-1",
        "learner_id": "learner-1",
        "claim_text": "Student repeatedly drops negative signs in two-step equations.",
        "construct_type": ConstructType.MISCONCEPTION,
        "construct_key": "linear_equations.negative_sign_transfer",
        "source_channel": SourceChannel.TUTOR_INFERENCE,
        "source_evidence_ids": ["turn-3"],
        "inference_level": InferenceLevel.LOCAL_INFERENCE,
        "scope_granularity": ScopeGranularity.SKILL,
        "scope_domain": "algebra",
        "confidence": 0.62,
    }
    data.update(overrides)
    return LearnerStateClaim(**data)


def test_claim_requires_construct_key_for_revision_matching():
    claim = make_claim()

    assert claim.construct_type is ConstructType.MISCONCEPTION
    assert claim.construct_key == "linear_equations.negative_sign_transfer"
    assert claim.admission_status is AdmissionStatus.CANDIDATE


def test_claim_rejects_blank_construct_key():
    with pytest.raises(ValidationError):
        make_claim(construct_key="   ")


def test_confidence_and_belief_are_range_checked():
    with pytest.raises(ValidationError):
        make_claim(confidence=1.5)

    with pytest.raises(ValidationError):
        BeliefState(p_mastery=0.4, variance=-0.1)


def test_claim_timestamps_are_timezone_aware():
    claim = make_claim()

    assert claim.created_at.tzinfo is not None
    assert claim.updated_at.tzinfo is not None
    assert claim.created_at <= datetime.now(timezone.utc)
