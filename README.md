# CLEAR-Mem

CLEAR-Mem is a reference implementation of claim-level regulation for
persistent learner-state memory in LLM tutoring systems. It represents each
stored learner judgment as an independently examinable claim and controls:

1. whether the claim may enter persistent memory;
2. how later evidence confirms, challenges, narrows, or supersedes it;
3. whether the claim may support a proposed instructional action; and
4. what evidence and decisions are retained for review.

CLEAR-Mem is not a complete tutoring system. The host system remains
responsible for interpreting learner performance, proposing learner-state
interpretations, planning instruction, and generating responses. CLEAR-Mem
operates between those host processes and persistent memory.

## Status

Version 0.1.0 is the first private release candidate. The core implementation
passes its isolated test suite and includes public-facing documentation,
licensing, and citation metadata. The remaining checks before public release
are tracked in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

## Quick Start

```bash
python3 -m examples.basic_workflow
```

The example admits an evidence-linked misconception claim and then checks
whether that claim may support corrective feedback.

## Tests

```bash
python3 -m pytest
```

The public candidate contains focused tests for claim validation, admission,
revision, instructional-use authorization, audit records, policy schemas, and
end-to-end governor behavior.

## Repository Layout

```text
clear/
  gates/       admission and instructional-use authorization
  update/      evidence-driven claim revision
  models/      claims, policies, decisions, and audit schemas
  crosscut/    audit and cross-cutting checks
  llm/         optional model adapters
examples/      minimal integration examples
tests/         isolated core test suite
docs/          architecture and integration boundaries
prompts/       prompt sets used in the reported generation and judgment stages
```

## Design Boundary

The public Python API is an alpha reference implementation. Machine-readable
field and enum names support execution and should not be treated as a complete
educational ontology. The paper's reader-facing concepts are learner-state
claims, learning evidence, observation conditions, applicability boundaries,
instructional-use authorization, audit, and teacher review.

## Related Benchmark

[LSS-Bench](https://github.com/AugustRemilia/LSS-Bench) provides the companion
benchmark for testing whether learner-state memory causes educationally
unjustified changes while preserving valid personalization.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) to cite this software
release. The accompanying paper will be added as a preferred citation after
publication metadata are available.

## License

CLEAR-Mem is licensed under the
[Apache License 2.0](LICENSE).
