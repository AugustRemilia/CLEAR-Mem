# Architecture

CLEAR-Mem receives a structured learner-state claim from a host tutoring
system. The claim links one learner-state interpretation to its supporting
evidence and observation conditions.

The runtime sequence is:

1. **Admission:** evaluate whether the evidence supports persistent storage and
   determine the claim's initial availability and permitted uses.
2. **Revision:** compare later evidence with active claims for the same learner
   and learning target, then confirm, challenge, supersede, or retire them.
3. **Context selection:** retrieve claims that remain applicable to the current
   learner, learning target, evidence, and instructional context.
4. **Use authorization:** decide whether a selected claim may support the
   proposed instructional action.
5. **Audit and review:** record explicit system decisions and refer unresolved
   claims or proposed uses for teacher review.

The `Governor` is the side-effect boundary. Admission and authorization gates
return decisions; the governor applies accepted state changes to the in-memory
store and appends corresponding audit events.

## Host Responsibilities

The host tutoring system:

- interprets tasks and learner responses;
- records observable learning events;
- proposes learner-state interpretations;
- constructs teaching plans and responses; and
- presents review material to a teacher when required.

CLEAR-Mem does not inspect hidden model reasoning. It regulates explicit
learner-state records and their observable instructional uses.
