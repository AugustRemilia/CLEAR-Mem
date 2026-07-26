"""Policy schema and experimental condition mapping."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clear.types import ConditionId, UseType


DEFAULT_ADMISSION_CRITERIA = [
    "schema_validity",
    "learner_specificity",
    "construct_clarity",
    "evidence_grounding",
    "current_consistency",
    "scope_appropriateness",
    "temporal_validity",
    "governance_admissibility",
]


class Policy(BaseModel):
    """Configuration surface for CLEAR conditions and thresholds."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    policy_version: str = "clear-schema-batch1-v1"
    condition_id: ConditionId | None = None
    condition_family: Literal["baseline", "clear_ablation"] = "clear_ablation"
    baseline_mode: Literal["none", "no_governance", "prompt_only", "memory_maintenance"] = "none"
    enabled_criteria: list[str] = Field(default_factory=lambda: list(DEFAULT_ADMISSION_CRITERIA))
    construct_policy_required: bool = True
    low_impact_uses: list[UseType] = Field(
        default_factory=lambda: [UseType.FEEDBACK_WORDING, UseType.EXAMPLE_SELECTION]
    )
    high_impact_uses: list[UseType] = Field(
        default_factory=lambda: [
            UseType.SCAFFOLD_LEVEL,
            UseType.SUPPORT_REDUCTION,
            UseType.CHALLENGE_ESCALATION,
            UseType.DIFFICULTY_ADAPTATION,
            UseType.LEARNING_PATH_RECOMMENDATION,
            UseType.TEACHER_ALERT,
            UseType.ABILITY_LABEL,
        ]
    )
    high_impact_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    z_lower_bound: float = Field(default=1.0, ge=0.0)
    reconfirm_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.05, ge=0.0)
    slip: float = Field(default=0.10, ge=0.0, le=1.0)
    guess: float = Field(default=0.20, ge=0.0, le=1.0)
    init_p_mastery: float = Field(default=0.30, ge=0.0, le=1.0)
    init_variance: float = Field(default=0.04, ge=0.0)
    misconception_damping: float = Field(default=0.50, ge=0.0, le=1.0)
    supersede_delta: float = Field(default=0.25, ge=0.0, le=1.0)
    supersede_min_evidence: int = Field(default=2, ge=1)
    structured_adjudication: bool = True
    use_bayesian: bool = True
    use_construct_router: bool = True
    llm_judge_enabled: bool = False

    @field_validator("enabled_criteria")
    @classmethod
    def validate_enabled_criteria(cls, values: list[str]) -> list[str]:
        unknown = sorted(set(values) - set(DEFAULT_ADMISSION_CRITERIA))
        if unknown:
            raise ValueError(f"unknown admission criteria: {unknown}")
        return values

    @classmethod
    def for_condition(cls, condition_id: ConditionId | str) -> "Policy":
        condition = ConditionId(condition_id)
        if condition is ConditionId.B0:
            return cls(
                condition_id=condition,
                condition_family="baseline",
                baseline_mode="no_governance",
                structured_adjudication=False,
                use_bayesian=False,
                use_construct_router=False,
            )
        if condition is ConditionId.B1:
            return cls(
                condition_id=condition,
                condition_family="baseline",
                baseline_mode="prompt_only",
                structured_adjudication=False,
                use_bayesian=False,
                use_construct_router=False,
            )
        if condition is ConditionId.B2:
            return cls(
                condition_id=condition,
                condition_family="baseline",
                baseline_mode="memory_maintenance",
                structured_adjudication=False,
                use_bayesian=False,
                use_construct_router=False,
            )
        if condition is ConditionId.C0:
            return cls(
                condition_id=condition,
                baseline_mode="no_governance",
                structured_adjudication=False,
                use_bayesian=False,
                use_construct_router=False,
            )
        if condition is ConditionId.C1:
            return cls(
                condition_id=condition,
                structured_adjudication=True,
                use_bayesian=False,
                use_construct_router=False,
            )
        if condition is ConditionId.C2:
            return cls(
                condition_id=condition,
                structured_adjudication=True,
                use_bayesian=True,
                use_construct_router=True,
            )
        if condition is ConditionId.C2A:
            return cls(
                condition_id=condition,
                structured_adjudication=True,
                use_bayesian=True,
                use_construct_router=False,
            )
        raise ValueError(f"unsupported condition: {condition_id}")
