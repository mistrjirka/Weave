"""Your next-token training recipe. Run: python train.py --config config.json.

Edit this loss or replace the script with your own PyTorch loop. Data loading,
accumulation, device placement, evaluation and checkpoints live in weave_support.
"""

from torch.nn import functional as F
from weave_support.text_training import main


def compute_loss(logits, targets):
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.to(logits.device).reshape(-1),
        ignore_index=-100,
        reduction="sum",  # The trainer applies token weighting across batches/GPUs.
    )


if __name__ == "__main__":
    main(loss_fn=compute_loss)
