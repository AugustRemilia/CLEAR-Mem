# Source Provenance

This candidate was extracted from `CLEAR/implementation` on 2026-07-26.

Included:

- the `clear` core package, excluding `clear.experiments`;
- focused tests for schemas, admission, revision, authorization, audit, policy,
  and governor integration; and
- a new public-facing example and documentation.

Excluded:

- internal experiment generators;
- 225 MB of experiment reports and raw outputs;
- caches and compiled Python files;
- provider-specific runtime notes; and
- manuscript, review, and planning files.

One former test of internal experiment runners was replaced with a repository
boundary test that verifies the public core package does not import benchmark
or internal experiment modules.
