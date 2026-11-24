"""Managed text trainer: causal text / assistant-only tool-call SFT.

Run: python train.py --config config.json
Multi-GPU: python -m torch.distributed.run --standalone --nproc_per_node=2 train.py --config config.json
The user-owned train.py supplies the loss function; custom loops can use Run directly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from model import Model
from weave_training import (
    Run,
    atomic_checkpoint_path,
    atomic_json,
    file_hash,
    available_cpu_threads,
)


from text_training_data import TokenDataset, load_rows, prepare_data, batch, language_loss
from text_training_support import move, place_model, checkpoint_state, capture_rng, gather_rng, restore_rng, evaluate
from text_training_loop import main


if __name__ == "__main__":
    main()
