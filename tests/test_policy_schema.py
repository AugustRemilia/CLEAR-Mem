import pytest

from clear.models.policy import Policy
from clear.types import ConditionId, UseType


def test_condition_mapping_c0_no_governance_baseline():
    policy = Policy.for_condition("C0")

    assert policy.condition_id is ConditionId.C0
    assert policy.condition_family == "clear_ablation"
    assert policy.baseline_mode == "no_governance"
    assert policy.structured_adjudication is False
    assert policy.use_bayesian is False
    assert policy.use_construct_router is False


def test_external_baseline_condition_mapping():
    b0 = Policy.for_condition("B0")
    b1 = Policy.for_condition("B1")
    b2 = Policy.for_condition("B2")

    assert b0.condition_id is ConditionId.B0
    assert b0.condition_family == "baseline"
    assert b0.baseline_mode == "no_governance"
    assert b0.structured_adjudication is False

    assert b1.condition_id is ConditionId.B1
    assert b1.condition_family == "baseline"
    assert b1.baseline_mode == "prompt_only"
    assert b1.structured_adjudication is False

    assert b2.condition_id is ConditionId.B2
    assert b2.condition_family == "baseline"
    assert b2.baseline_mode == "memory_maintenance"
    assert b2.structured_adjudication is False


def test_condition_mapping_c2_full_clear():
    policy = Policy.for_condition(ConditionId.C2)

    assert policy.condition_id is ConditionId.C2
    assert policy.condition_family == "clear_ablation"
    assert policy.baseline_mode == "none"
    assert policy.structured_adjudication is True
    assert policy.use_bayesian is True
    assert policy.use_construct_router is True


def test_condition_mapping_c2a_uniform_scalar_confidence_update():
    policy = Policy.for_condition("C2a")

    assert policy.condition_id is ConditionId.C2A
    assert policy.structured_adjudication is True
    assert policy.use_bayesian is True
    assert policy.use_construct_router is False


def test_construct_policy_is_required_before_low_impact_use():
    policy = Policy()

    assert policy.construct_policy_required is True
    assert UseType.FEEDBACK_WORDING in policy.low_impact_uses
    assert UseType.SCAFFOLD_LEVEL in policy.high_impact_uses


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        Policy.for_condition("C3")


def test_unknown_admission_criterion_is_rejected():
    with pytest.raises(ValueError):
        Policy(enabled_criteria=["schema_validity", "typo_criterion"])
