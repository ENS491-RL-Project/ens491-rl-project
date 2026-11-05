---
paths:
  - "src/task_detection/**"
  - "src/options/task_lifecycle.py"
  - "tests/test_autoencoder.py"
  - "tests/test_gru.py"
---

# Task detection (AE + GRU)

- Interface contracts: AE and GRU sections of `docs/ENS491_Module_Interfaces.md`; design in v2 §6.
- Autoencoder: MLP `147 → 64 → 32 → 64 → 147` on input normalized with the same `ChannelNormalize` utility as the policies (object /10, color /5, state /2).
- GRU task identifier: hidden size 64, sequence length N = 20 by default, supervised training. Its task id selects a root node through the task registry.
