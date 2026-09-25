# Runtime configuration as run


Taken from the run logs of the reported experiments. Values are reported per
component and stage; library defaults are noted where the run logs carry no
persistent field.

| Item | Single-turn | 24-turn test | 16-turn coverage | 16-turn stress |
|---|---|---|---|---|
| Completion cap (tokens per call) | 512 | 512 | 640 | 768 |
| Client-side prompt budget (tokens) | 5,600 | default 4,500 | default 4,500 | default 4,500 |
| Judgment prompt budget (tokens) | - | default 5,500 | default 5,500 | default 5,500 |
| temperature | 0 | 0 | 0 | 0 |
| top-p / top-k / seed | not set | not set | not set | not set |
| Reasoning traces in judgment | - | disabled | disabled | disabled |
| Backend | mimo-v2.5-pro | mimo-v2.5-pro | mimo-v2.5-pro | mimo-v2.5-pro |
| Structured output | JSON object | JSON object | JSON object | JSON object |

The multi-turn prompt-budget values are the script defaults; the run logs carry
no persistent field for them, so they are reported as defaults rather than as
verified run values. No repeated runs were performed (see Section IV-C of the
paper).
