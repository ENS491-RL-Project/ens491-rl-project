# 0010 — Design decisions are settled by experiment, and every decision links to its experiments

Status: Accepted
Date: 2026-11-12
Reviewed by Emir: yes — his rule, 2026-11-12
Source: Emir, 2026-11-12

## Decision
- A design question is settled by trying the alternatives and keeping what works, not by asking
  the supervisor for an answer.
- Every experiment is recorded (`results.md`, linking its plan and result files), and every decision
  file lists the experiments behind it under **Evidence**. "None yet" is allowed but must name the
  experiment that would settle it.
- The supervisor is consulted on scope, framing and direction, and is shown the evidence.

## Why
Several v2 choices were made by reasoning or by default during fast autonomous runs and were never
tested against alternatives (0004, 0005, 0009 before review). The project already lists such
questions as empirical ([`open-questions.md`](../open-questions.md), last section).

## Evidence
Not applicable (process rule).

## Affects
Every decision file (Evidence section), [`open-questions.md`](../open-questions.md) (how-to-settle column), `results.md`.
