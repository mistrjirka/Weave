# Design assistance and correctness checks

Weave distinguishes **errors** (a contract cannot execute as declared) from
**design notes** (a valid graph deserves a second look). Notes appear in the
validation drawer and the affected block's properties. Clicking the block path
opens its containing graph. Notes never prevent export or rewrite a graph.
They are not a prediction of training quality.

## Evidence and implemented rules

| Rule | Evidence checked in the graph | Suggested check and exceptions |
| --- | --- | --- |
| Very narrow hidden bottleneck | A linear branch compresses by at least 16× to 1–4 features, then expands to at least its original width along resolved unary linear/activation/norm/scale nodes. Alternatively, an FFN has an equally narrow internal hidden width. | Check whether the compression is intentional; compare wider versions on validation data. Final scalar outputs are excluded. Traversal stops at merges and containers. A narrow branch can be an intentional adapter; a bypass may preserve information. |
| Narrow feed-forward hidden width | An FFN's resolved hidden width is at most half the input width. Extreme cases use the existing bottleneck note instead. | This is a valid capacity and cost tradeoff. Consider comparing a wider FFN; a residual connection can preserve information outside this branch. The 1/2 threshold is a product heuristic, not a proven minimum. |
| Consecutive linear layers | A directly connected chain of linear projections, with one note per uninterrupted chain. A vocabulary head can terminate the chain. | Insert an activation if nonlinear depth was intended. Factorized linear maps can be useful and should remain editable. |
| Output softmax | A softmax directly feeds an output. | **If** training with PyTorch cross entropy, expose raw logits. Probability outputs are legitimate for inference and other losses. For `[B,T,V]`, flatten to `[B*T,V]` or move the class axis. |
| Unused learned branch | A resolved parameterized node has no directed path to any output in its scope. | Connect it if intended for training, or retain it as a draft. Auxiliary outputs count as real outputs. Shared weights may still train through another application. |
| Discrete token selection | A live token selection node returns integer token IDs. | Use the path for sampling; train a predictor using floating-point logits before selection. For masked diffusion, choose the loss positions deliberately. |
| Fixed zero scale | A live scale node has an exactly zero factor. | For finite inputs, this branch is silenced and has zero input gradients. A bypass may still carry gradients. Intentional ablations remain valid; small nonzero factors do not trigger the note. |
| Empty custom attention rows | A resolved custom boolean pattern has rows containing only False. | Lists zero-based query indices. Allow an intended key if the empty row is accidental, or retain intentional masking. A later union can supply keys, so this is local advice about the pattern, not a conclusion about every attention operation using it. |

The **16× and 1–4 thresholds are conservative product heuristics**, not research
claims or minimum architecture requirements. No width ratio can determine model
quality without the task, data, objective and experiments. Rules operate on the
same graph representation for all architectures, including nested blocks and loops.
They do not execute user Python.

Primary references researched for these choices:

- [PyTorch Linear](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html)
  defines the affine operation. By composing its formula, consecutive affine
  maps remain affine; a linear projection through a smaller dimension also
  limits rank on that branch. These are mathematical deductions, not empirical
  claims that deeper linear or low-rank models are always bad.
- [Deep Learning, chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
  discusses nonlinear hidden layers and architecture selection using validation.
- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)
  specifies unnormalized logits and the class-axis convention.
- [Scaled dot-product attention](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)
  defines the boolean mask convention: True allows a key to participate.
  Empty-row detection here checks explicit custom matrices, not arbitrary
  input-dependent or composed masks.
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.9/notes/autograd.html)
  explains gradients through the executed operation graph.
  [Automatic differentiation tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  explains differentiable floating-point computation and gradients.
- [Going Deeper with Convolutions, section 4](https://arxiv.org/html/1409.4842v1)
  uses **1×1 convolutions** for useful channel reduction. A 1×1 spatial kernel
  does not mean a representation has only one channel. We do not warn against
  1×1 convolutions.
- [LoRA](https://arxiv.org/abs/2106.09685) provides an important counterexample
  to blanket criticism of low-rank projections: constrained updates can be
  useful when the original computation is preserved.

## Checks deliberately left contextual

No universal warning requires causal attention, residuals, normalization,
dropout, a particular FFN ratio, or a particular depth. Bidirectional attention
is appropriate for encoders and masked diffusion. A future-token-leak warning
would require an explicit training objective and knowledge of the data/target
alignment; a graph name or the presence of a vocabulary head is insufficient.
Likewise, GPU memory and overfitting cannot be diagnosed from parameter count
alone. Custom Python contracts are declarations, not proofs of runtime behavior.

## Verification

Run `.venv/bin/python -m pytest tests -q` for the Python suite. It includes:

- Exported packages executed in separate Python processes, without importing the
  backend; forward inference and small fixed-batch training runs.
- Cross-entropy loss reduction, finite parameter gradients, optimizer updates,
  checkpoint reload, and graph parameter count agreement.
- An independently handwritten residual/shared-loop calculation compared with
  generated values and gradients, plus numerical finite-difference `gradcheck`.
- Future-token perturbations that must not change earlier causal logits.
- Attention, sparse masks, n-grams, shared/unshared loops, diffusion sampling,
  upstream architecture equation comparisons, and custom-source isolation.
- Invalid dimensions, unresolved bindings, missing connections, residual widths,
  attention mask dimensions/dtypes, matrix operand dtype, and node/port location.
- Advice triggering and suppression cases, valid scalar outputs, nonlinear
  layers, multiple outputs, nested scopes, and export despite warnings.

Run `npx playwright test tests/browser/advice.spec.ts` for browser interactions,
desktop/phone readability, node navigation, errors, repair, and export with notes.
Other browser and geometry suites cover the editor and all example scopes.
These checks establish specific contracts and regressions; they cannot prove all
possible user graphs, model quality, or untested GPU kernel behavior.
