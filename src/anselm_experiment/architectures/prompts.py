"""Shared prompt fragments — kept identical across architectures so the
ITER vs MULTI-PIPE vs MULTI-VOTE comparison measures architecture, not wording.
"""
from __future__ import annotations


TAG_VOCABULARY = """The schema defines a `tags` field on each step. Constraints reference these
tags (NOT step IDs). Apply tags to the steps that semantically perform the
tagged role:
- `creates_order`   — step where the customer order is created
- `raises_hold`     — step where the credit hold is raised
- `reviews_hold`    — step where a held order is reviewed (must declare sla_hours)
- `releases_hold`   — step where the hold is released (must produce audit fields)
- `rejects_hold`    — step where the hold is rejected (must produce audit fields)
- `escalates`       — step where a case is escalated to higher authority
- `closes_case`     — step where the case is closed"""


ARCHITECT_SYSTEM = (
    "You are a senior business-process architect. You design a process under "
    "an explicit constraint set and output the result as a JSON object that "
    "conforms to the provided schema.\n\n"
    + TAG_VOCABULARY
    + "\n\nOutput ONLY a JSON object. Do not include prose outside the JSON."
)
