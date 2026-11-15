# ens491-rl-project

ENS491 Graduation Project (#539): continual task acquisition with reduced catastrophic forgetting
via hierarchical policy reuse and interpretable decision-making (incl. GUI).

A hierarchical continual RL agent on [MiniGrid](https://minigrid.farama.org/). Every learned task
becomes a frozen, reusable **skill node**; composite nodes select among earlier frozen nodes as
options, to any depth. Freezing prevents forgetting; an autoencoder (novelty) and a GRU (task id)
decide when a task is new and which known task is running.
Curriculum: Unlock, DoorKey, UnlockPickup, BlockedUnlockPickup.

## Setup

Python 3.10, in a conda env named `ens491`:

```bash
conda create -n ens491 python=3.10
conda activate ens491
pip install -r requirements.txt
```

## Run

```bash
pytest tests/ -v                 # all tests
streamlit run src/gui/app.py     # GUI
```

Experiments are `scripts/train_*.py` and `scripts/eval_*.py`; frozen checkpoints and result JSONs
are in `runs/v2/`.

## Docs

- Start here: [`docs/kb/README.md`](docs/kb/README.md)
- Where the project stands: [`docs/kb/status.md`](docs/kb/status.md); measurements: [`docs/kb/results.md`](docs/kb/results.md)
- Reference design: [`docs/ENS491_Target_Architecture_v2.md`](docs/ENS491_Target_Architecture_v2.md)
- Decisions and open questions: [`docs/kb/decisions/`](docs/kb/decisions/), [`docs/kb/open-questions.md`](docs/kb/open-questions.md)
