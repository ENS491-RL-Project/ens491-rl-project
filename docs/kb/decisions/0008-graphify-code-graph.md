# 0008 — graphify is the team's code map; the graph is built locally, not committed

Status: Accepted
Date: 2026-10-25 (decision) · 2026-11-05 (PR #13 merged)   Approved by: Team (PR #13 merged)   Source: PR #13; CLAUDE.md §graphify; PR #19 (pinning, merged 2026-11-12)
Reviewed by Emir: yes — PR #13 review

## Decision
Use graphify for code navigation. `graphify-out/` is generated per machine and stays out of git.

## Why
- Team agreement to share one code map.
- Committing the graph produced multi-MB diffs and machine-local timestamps (review of PR #13).

## Rejected alternatives
- Commit `graphify-out/`: rejected in review.

## Affects
CLAUDE.md; [`requirements-dev.txt`](../../../requirements-dev.txt) since PR #19. graphify maps code only; reasons and research live in [`docs/kb/`](../).
