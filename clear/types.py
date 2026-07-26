"""Shared enums for CLEAR schemas."""

from __future__ import annotations

from enum import StrEnum


class ConstructType(StrEnum):
    KNOWLEDGE_STATE = "knowledge_state"
    MISCONCEPTION = "misconception"
    PARTIAL_MASTERY = "partial_mastery"
    STRATEGY = "strategy"
    HELP_SEEKING = "help_seeking"
    AFFECTIVE_STATE = "affective_state"
    LANGUAGE_NEED = "language_need"
    ACCESSIBILITY_NEED = "accessibility_need"
    IDENTITY_OR_SENSITIVE = "identity_or_sensitive"
    PREFERENCE = "preference"
    BEHAVIOR_PATTERN = "behavior_pattern"


class SourceChannel(StrEnum):
    EXPLICIT_STATEMENT = "explicit_statement"
    SYSTEM_AUTO_SAVE = "system_auto_save"
    TUTOR_INFERENCE = "tutor_inference"
    COMPACTION_SUMMARY = "compaction_summary"
    CROSS_SESSION_SYNTHESIS = "cross_session_synthesis"
    TEACHER_NOTE = "teacher_note"
    EXTERNAL_RECORD = "external_record"


class InferenceLevel(StrEnum):
    OBSERVATION = "observation"
    LOCAL_INFERENCE = "local_inference"
    CROSS_TASK_INFERENCE = "cross_task_inference"
    GLOBAL_PROFILE_CLAIM = "global_profile_claim"


class ScopeGranularity(StrEnum):
    TURN = "turn"
    PROBLEM = "problem"
    SKILL = "skill"
    UNIT = "unit"
    CROSS_SESSION = "cross_session"


class TemporalStatus(StrEnum):
    CURRENT = "current"
    RECENT = "recent"
    STALE = "stale"
    EXPIRED = "expired"
    CONTESTED = "contested"


class AdmissionStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    WEAK_CONTEXT = "weak_context"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    AUDIT_ONLY = "audit_only"


class ReviewStatus(StrEnum):
    NONE = "none"
    TEACHER_REVIEW_NEEDED = "teacher_review_needed"
    TEACHER_APPROVED = "teacher_approved"
    TEACHER_REJECTED = "teacher_rejected"


class UseType(StrEnum):
    FEEDBACK_WORDING = "feedback_wording"
    EXAMPLE_SELECTION = "example_selection"
    CORRECTIVE_FEEDBACK = "corrective_feedback"
    SCAFFOLD_LEVEL = "scaffold_level"
    SUPPORT_REDUCTION = "support_reduction"
    CHALLENGE_ESCALATION = "challenge_escalation"
    DIFFICULTY_ADAPTATION = "difficulty_adaptation"
    LEARNING_PATH_RECOMMENDATION = "learning_path_recommendation"
    TEACHER_ALERT = "teacher_alert"
    ABILITY_LABEL = "ability_label"


class ImpactTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RevisionRegime(StrEnum):
    QUANTIFIABLE_KNOWLEDGE = "quantifiable_knowledge"
    BEHAVIORAL_PREFERENCE = "behavioral_preference"
    SENSITIVE_NEED = "sensitive_need"


class ConditionId(StrEnum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C2A = "C2a"


class AuditEventType(StrEnum):
    INTAKE = "intake"
    ADMISSION = "admission"
    USE_AUTHORIZATION = "use_authorization"
    REVISION = "revision"
    DECAY = "decay"
    FAIRNESS_BLOCK = "fairness_block"
    TEACHER_REVIEW = "teacher_review"


class RevisionAction(StrEnum):
    CONFIRM = "confirm"
    CHALLENGE = "challenge"
    SUPERSEDE = "supersede"
    REJECT_OVERRIDE = "reject_override"
    DECAY_EXPIRE = "decay_expire"
    NO_OP = "no_op"


class Observation(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NEUTRAL = "neutral"
