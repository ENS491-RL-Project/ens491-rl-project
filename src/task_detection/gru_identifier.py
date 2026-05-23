from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.types import GRUOutput, Observation, ObsSequence, TaskID


def _to_tensor(obs: Observation | object) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.float()
    import numpy as np
    return torch.tensor(obs, dtype=torch.float32)


class GRUTaskIdentifier(nn.Module):
    """
    GRU-based task classifier: "Which task is this sequence from?"

    Input: last SEQ_LEN observations (zero-padded if buffer not full).
    Output: GRUOutput with task_id (-1 if confidence < threshold), confidence,
            and full probability vector.

    register_new_task() expands the output head to cover the new task ID.
    It is idempotent: no-op if the head already covers that task.

    Training protocol for demo:
        1. Collect labelled sequences offline (one pass per known task).
        2. Call train_supervised(sequences, labels).
        3. Validate task_id / confidence / all_probs / register_new_task() head expansion.
    """

    SEQ_LEN = 20
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        obs_dim: int = 147,
        hidden_size: int = 64,
        num_tasks: int = 1,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self._num_tasks = num_tasks

        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_tasks)

        self._obs_buffer: deque[torch.Tensor] = deque(maxlen=self.SEQ_LEN)
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Rolling buffer helpers
    # ------------------------------------------------------------------

    def observe(self, obs: Observation) -> None:
        """Append one observation to the rolling buffer."""
        self._obs_buffer.append(_to_tensor(obs))

    def get_sequence(self) -> ObsSequence:
        """Return (SEQ_LEN, obs_dim) tensor, zero-padded at start if buffer not full."""
        buf = list(self._obs_buffer)
        pad_len = self.SEQ_LEN - len(buf)
        tensors = [torch.zeros(self.obs_dim)] * pad_len + buf
        return torch.stack(tensors)  # (SEQ_LEN, obs_dim)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def forward(self, obs_sequence: ObsSequence) -> GRUOutput:
        """
        obs_sequence: (N, obs_dim) — typically N = SEQ_LEN.
        Returns GRUOutput. task_id = -1 when confidence < CONFIDENCE_THRESHOLD.
        """
        x = obs_sequence.unsqueeze(0)  # (1, N, obs_dim)
        self.eval()
        with torch.no_grad():
            _, h_n = self.gru(x)          # h_n: (1, 1, hidden_size)
            logits = self.head(h_n.squeeze())  # (num_tasks,)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max().item()
        predicted = probs.argmax().item()
        task_id: TaskID = predicted if confidence >= self.CONFIDENCE_THRESHOLD else -1
        return GRUOutput(task_id=task_id, confidence=confidence, all_probs=probs.detach())

    def register_new_task(self, task_id: TaskID) -> None:
        """
        Expand the output head to cover task_id.
        Idempotent: does nothing if head already has task_id + 1 outputs.
        """
        needed = task_id + 1
        if needed <= self._num_tasks:
            return  # head already covers this task

        old_head = self.head
        new_head = nn.Linear(self.hidden_size, needed)
        with torch.no_grad():
            new_head.weight[:self._num_tasks] = old_head.weight
            new_head.bias[:self._num_tasks] = old_head.bias
            # New row: small random init
            nn.init.xavier_uniform_(new_head.weight[self._num_tasks:])
            nn.init.zeros_(new_head.bias[self._num_tasks:])
        self.head = new_head
        self._num_tasks = needed
        # Rebuild optimiser to include new head parameters
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._optimizer.defaults["lr"])

    def train_supervised(
        self,
        sequences: torch.Tensor,  # (B, SEQ_LEN, obs_dim)
        labels: torch.Tensor,     # (B,) int64
        epochs: int = 10,
    ) -> float:
        """Supervised cross-entropy training. Returns final epoch loss."""
        self.train()
        final_loss = 0.0
        for _ in range(epochs):
            self._optimizer.zero_grad()
            _, h_n = self.gru(sequences)          # h_n: (1, B, hidden_size)
            logits = self.head(h_n.squeeze(0))    # (B, num_tasks)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            self._optimizer.step()
            final_loss = loss.item()
        self.eval()
        return final_loss

    @property
    def num_tasks(self) -> int:
        return self._num_tasks
