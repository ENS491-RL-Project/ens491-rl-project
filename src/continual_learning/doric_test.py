"""
doric_test.py
Minimal Progressive Network using Doric library.
Goal: 2 columns, lateral connections, MiniGrid-Empty-8x8 compatible obs shape.
 
Key questions being evaluated:
  1. How much boilerplate does Doric require?
  2. Does ProgColumnGenerator cleanly handle new column addition?
  3. Is freeze/gradient isolation working correctly?
  4. How easily can this extend to recursive/unbounded depth?
"""

import torch
import torch.nn as nn
from Doric import (
    ProgNet,
    ProgColumn,
    ProgColumnGenerator,
    ProgDenseBlock,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OBS_SIZE   = 148   # MiniGrid-Empty-8x8 with FlatObsWrapper
HIDDEN     = 64
N_ACTIONS  = 7     # MiniGrid discrete action space
 
 
# ---------------------------------------------------------------------------
# Column Generator
# ---------------------------------------------------------------------------

class MiniGridColumnGenerator(ProgColumnGenerator):
    """
    Generates a new ProgColumn for a new MiniGrid task.
    numLaterals = number of parent coumns (handled automatically by ProgNet).
    """

    def __init__(self, obs_size, hidden, n_actions):
        self.obs_size = obs_size
        self.hidden = hidden
        self.n_actions = n_actions

    def generateColumn(self, parentCols, msg=None):
        n_lat = len(parentCols)
        col_id = msg if msg else f"tas{n_lat+1}"

        blocks = [
            ProgDenseBlock(self.obs_size, self.hidden, numLaterals=n_lat),
            ProgDenseBlock(self.hidden, self.hidden, numLaterals=n_lat),
            ProgDenseBlock(self.hidden, self.n_actions, numLaterals=n_lat, activation=None),
        ]

        return ProgColumn(colID=col_id, blockList=blocks, parentCols=parentCols)
    


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def verify_freeze(net, frozen_id, active_id):
    """Check that frozen column really has no gradients after backward."""
    x = torch.rand(1, OBS_SIZE)
    out = net.forward(active_id, x)
    loss = out.sum()
    loss.backward()

    frozen_col = net.getColumn(frozen_id)
    active_col = net.getColumn(active_id)

    frozen_has_grad = any(p.grad is not None for p in frozen_col.parameters())
    active_has_grad = any(p.grad is not None for p in active_col.parameters())

    return (not frozen_has_grad), active_has_grad # (frozen_ok, active_ok)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("Doric Progressive Network -- Minimal Test")
    print("=" * 55)

    gen = MiniGridColumnGenerator(OBS_SIZE, HIDDEN, N_ACTIONS)
    net = ProgNet(colGen=gen)

    # -----------------------------------------------------------------------
    # Step 1: Add column for Task 1
    # -----------------------------------------------------------------------

    id1 = net.addColumn(msg="task1")
    print(f"\n[+] Added column: {id1}")

    x = torch.randn(4, OBS_SIZE)
    out1 = net.forward(id1, x)
    print(f"    forward({id1}) output shape: {out1.shape}")
    
    t, tr = count_params(net.getColumn(id1))
    print(f"    params total={t} trainable={tr}")

    # Simulate: Task 1 is done training --> freeze
    net.freezeColumn(id1)
    print(f"    frozen: {net.isColumnFrozen(id1)}")

    # -----------------------------------------------------------------------
    # Step 2: Add column for Task 2 (lateral from Task 1)
    # -----------------------------------------------------------------------

    id2 = net.addColumn(msg="task2")
    print(f"\n[+] Added column: {id2}")

    out2 = net.forward(id2, x)
    print(f"    forward({id2}) output shape: {out2.shape}")

    t, tr = count_params(net.getColumn(id2))
    print(f"    params total={t} trainable={tr}")

    # -----------------------------------------------------------------------
    # Step 3: Verify gradient isolation
    # -----------------------------------------------------------------------

    frozen_ok, active_ok = verify_freeze(net, id1, id2)
    print(f"\n[✓] Gradient isolation check:")
    print(f"    {id1} has NO grad after backward: {frozen_ok}")
    print(f"    {id2} HAS grad after backward: {active_ok}")

    # -----------------------------------------------------------------------
    # Step 4: Architecture summary
    # -----------------------------------------------------------------------

    print(f"\n[i] ProgNet data:")
    data = net.getData()
    for col in data["cols"]:
        print(f"    col={col['colID']}  rows={col['rows']}  "
              f"frozen={col['frozen']}  parents={col['parent_cols']}")

    # -----------------------------------------------------------------------
    # Step 5: Extensibility note — would a 3rd column work?
    # -----------------------------------------------------------------------

    net.freezeColumn(id2)
    id3 = net.addColumn(msg="task3")
    out3 = net.forward(id3,x)
    print(f"\n[+] Added 3rd column: {id3}")
    print(f"    forward({id3}) output shape: {out3.shape}")
    col3 = net.getColumn(id3)
    print(f"    num lateral connections per block: "
        f"{col3.blocks[0].numLaterals}")   # should be 2
    
    print("\n" + "=" * 55)
    print("DONE -- evaluate Doric vs custom PN on:")
    print(" - boilerplate (this file: ~120 lines total)")
    print(" - lateral count auto-scaling: YES (via parentCols)")
    print(" - freeze isolation: verified above")
    print(" - recursive extension: manually addColumn each time")
    print(" - CONSTRAINT: all columns must have same numRows")
    print("=" * 55)


if __name__ == "__main__":
    main()