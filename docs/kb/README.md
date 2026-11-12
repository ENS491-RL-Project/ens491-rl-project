# Knowledge base

Why the project is the way it is, and where it stands. The design itself is in
[`../ENS491_Target_Architecture_v2.md`](../ENS491_Target_Architecture_v2.md); the code map is graphify.

Read in this order:

1. [`status.md`](status.md) — where each track stands and what the open PRs close.
2. [`open-questions.md`](open-questions.md) — what is not decided yet, and what it blocks.
3. [`decisions/`](decisions/) — one file per decision: what, why, what was rejected, who approved.
4. [`results.md`](results.md) — every measurement, newest first, linked to its plan or JSON.
5. [`glossary.md`](glossary.md) — the project's terms.
6. `research/` — which papers each design element rests on. *(later PRs)*

Rules:

- One screen per file, except `research/`. Every line points to a source (PR, commit, plan, v2 section, meeting note).
- Decisions are settled by experiment; each decision file lists its experiments under **Evidence** ([0010](decisions/0010-decide-empirically.md)).
- KB changes go through PRs like code. A decision becomes `Accepted` only when Emir sets it.
- A PR that changes results or roadmap state updates [`status.md`](status.md) (and [`results.md`](results.md)) in the same PR.
- A change to a 🔒 rule or an open question starts as a `Proposed` decision file, before any code.
