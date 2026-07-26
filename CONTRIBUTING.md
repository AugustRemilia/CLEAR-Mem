# Contributing

This repository contains the version 0.1.0 release candidate.

Before proposing a change:

1. keep learner-state claims distinct from raw dialogue history;
2. preserve the separation between memory admission and instructional-use
   authorization;
3. add or update tests for every behavioral change;
4. avoid committing learner data, credentials, raw API logs, or model-provider
   outputs without an explicit redistribution decision; and
5. run `python3 -m pytest`.

Contributions are accepted under the repository's Apache License 2.0.
