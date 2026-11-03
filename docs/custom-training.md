# Custom training loops

Choose **Custom Python** when your script owns the training loop: RL, unusual
objectives, several optimizers, or an existing training library. Weave freezes the
graph and files, starts the script, records reports, and keeps run history. It
does not choose an RL algorithm, collect rollouts, execute tools, or define rewards.

The helpers below ship inside each new stage's `weave_support/`. They need no
separate Weave package. Edit `train.py` and your own modules; generated support
is refreshed when you save a stage. Existing frozen runs keep their code.

To begin RL from a pretrained model, create a **Custom Python** stage and choose
**New stage from checkpoint weights**. Load with `weights_only=True` as below;
your new optimizer, schedule and counters start fresh. Continuing the same RL
experiment instead requires the full resume path and all state it needs.

## Reporting without a Trainer

```python
from weave_support import Run

with Run() as run:
    run.capabilities(metrics=True, samples=True, commands=False, checkpoint=False)
    run.configure(
        {"devices": [0], "strategy": "single", "precision": "bf16"},
        metrics={
            "train/reward": {"label": "Mean reward"},
            "eval/success": {
                "label": "Task success", "direction": "max", "min": 0, "max": 1,
            },
        },
        primary_metric="eval/success",
        headline_metrics=["eval/success", "train/reward"],
        step_unit="policy updates",
    )
    # Your loop computes these values.
    run.log({"train/reward": mean_reward}, step=update,
            counters={"episodes": episodes, "rollouts": rollout_batches})
    run.progress(completed=update, total=planned_updates, unit="policy updates")
    run.sample(prompt=task, output=generated_call,
               metadata={"reward": reward, "policy_version": policy_version})
```

The variables in these snippets belong to your script. `configure()` records
**applied settings**; it does not move the model or change precision. Its metric
labels, bounds and step unit appear in the dashboard. The primary metric also
ranks checkpoints with matching scores. Use `direction="min"` for a quantity you
want to minimize. Report evaluation scores at the checkpoint's original step,
not at the current training step.

`step` is your primary update counter. Named counters are cumulative nonnegative
integers and can be selected as chart axes. For multiple optimizers, explicit
`run.log(..., step=update)` avoids confusing policy updates with individual
optimizer steps. A single-optimizer loop can still use `Run(optimizer=optimizer)`
and `run.log_loss(loss)` for automatic LR and step tracking.

Reports are written locally. A background sender delivers them when you launch
independently with a reporting connection. Network outages do not block the loop.
Local JSON serialization, file writes, checkpoint saves and tensor-to-scalar
conversions still have a cost: choose an appropriate logging interval.

## GUI launch settings

A custom script owns its compute configuration. Controls are enabled only for
fields the script explicitly declares and consumes:

```python
from weave_support.custom_training import launch_settings

WEAVE_SUPPORTED_SETTINGS = ("devices", "strategy", "precision")
settings = launch_settings({"devices": [], "strategy": "single", "precision": "fp32"})
```

For GUI launches, `settings` includes the frozen launch values. Independently
launched scripts use the defaults passed in Python. Reading this dictionary does
not implement AMP, compilation, distributed training, or evaluation. Declare a
field only after your loop uses it. Report the effective values with
`run.configure()` after applying them.

Supported names are `devices`, `strategy`, `precision`, `placement`, `compile_mode`,
`fused_optimizer`, `profile_steps`, `evaluation_mode`, and `evaluation_device`.
The declaration must be a literal list or tuple at the top level of `train.py` so
Weave can inspect it without executing the file. Your data, optimizer, reward,
loss and schedule remain Python code. Custom scripts do not automatically gain
the built-in Trainer's GPU tuning or live validation controls.

Weave passes `--config config.json` to custom scripts. If your script uses its own
argument parser, accept that argument or use `parse_known_args()`. The frozen file
is launch metadata; you do not need to maintain a user configuration JSON file.

## Checkpoints, safe stop and resume

```python
from weave_support.custom_training import Checkpointer

checkpointer = Checkpointer(
    run,
    components={"model": model, "optimizer": optimizer, "scheduler": scheduler},
    rank_local={"sampler": sampler},  # Optional objects with state_dict/load_state_dict.
    exact_resume=True,
)
if settings.get("checkpoint"):
    extra = checkpointer.load(
        settings["checkpoint"],
        weights_only=settings.get("start_mode") != "resume",
    )
run.capabilities(commands=True, checkpoint=True, resume=True)

# At a safe boundary, after the full update:
if run.handle_commands(checkpointer.save):
    break

# A normal scheduled save:
path = checkpointer.save(extra={"reward_version": "mobile-v2"})
```

Register any stateful components you need: policy/value models, named optimizers,
schedulers, AMP scaler, sampler, replay buffer, or environment. `rank_local`
saves separate state for each distributed rank. Shared components are saved from
rank zero. Components must expose `state_dict()` and `load_state_dict()`.

The helper atomically saves and registers the checkpoint, including Python,
PyTorch and optional NumPy RNG states, named counters, tokens, and extra state.
Default filenames are unique; explicit checkpoint filenames cannot be reused.
State must contain tensors and primitive dictionaries/lists/tuples, not pickled
application objects. For large replay buffers, external storage and your own
checkpoint implementation may be more appropriate.

**`exact_resume=True` is your declaration that all required state is covered.**
The helper cannot discover an unregistered environment, data iterator, reference
policy, reward version, or external service state. For fresh stochastic rollouts,
you may not need to preserve completed trajectories; restarting mid-rollout is a
different contract. Do not claim exact resume if the environment cannot restore.
Only advertise `resume=True` after implementing the load path.

The GUI offers pause/resume for local custom runs that declare support and publish
a checkpoint marked for exact resume. Each restart is a new frozen run with a
recorded script/checkpoint lineage. Choose the recorded or saved edited script.
Incompatible model tensor shapes fail visibly; fix the script and retry, or use
the previous script. User component loaders can still fail after restoring an
earlier component; retry in a fresh process rather than continuing that process.

The top-level `"model"` entry supports checkpoint chat when it matches the frozen
graph. Keep a training-only critic in a separate component, rather than changing
the chat model's state dictionary. Weave does not translate arbitrary external
checkpoint formats. You can keep saving your own checkpoints and register them
with `run.checkpoint(path, metadata=...)` instead.

## Tables and artifacts

```python
run.table("Tool rollouts", trajectories, metadata={
    "policy_version": checkpoint_hash,
    "reward_version": "mobile-v2",
    "tool_catalog_sha256": catalog_hash,
})
run.artifact(completed_file, name="Reward breakdown")
```

Rows are dictionaries; `trajectories` can be an iterator. Weave streams the full
table to JSONL and sends a bounded preview (at most 20 rows / 64 KB) to **Tables &
artifacts**. Include task, generated call, tool/argument correctness, reward, and
success as columns. Store provenance and simulator/reward/catalog versions in
metadata. Files must be completed and inside the run directory.

Large values belong in artifacts, not metric events. Individual table rows are
limited to 1 MB and 64 columns; previews shorten long cells. Download the full
file to inspect all content. Independent reporting sends previews and metadata,
not the artifact/checkpoint bytes; the UI marks files kept on the training host.

## Background evaluation

```python
from weave_support.custom_evaluation import AsyncEvaluator, EvaluationResult

# Define this at module scope; spawn must be able to import it.
def evaluate_policy(checkpoint_path, device):
    # Load your policy once per snapshot; cache models/tokenizers in this worker
    # if your evaluator can safely reuse them. Run held-out tasks here.
    return EvaluationResult(metrics={"eval/success": measured_success})

if __name__ == "__main__":
    # Inside the Run context, on rank zero:
    with AsyncEvaluator(run, evaluate_policy, device="cuda:1", timeout=600) as evaluator:
        # Inside your loop:
        evaluator.poll()
        if evaluation_due and not evaluator.busy:
            path = checkpointer.save()
            evaluator.submit(path, step=update)
```

Use `"cpu"` or `"cuda:N"` for a CUDA GPU. A single worker stays alive between jobs;
user code decides whether to cache its models or other resources. Each job uses a
completed immutable checkpoint. Results retain that checkpoint's step, hash, and
counters even if training has advanced. The checkpoint selector associates these
scores with the matching file/hash. Failures and timeouts appear in run logs and
evaluation status; a failed worker is replaced on the next submission.

There is one outstanding job and no growing queue. `submit()` returns `False`
while busy. `poll()` reports results without waiting for evaluation; exiting the
context waits for the outstanding job, up to its timeout. Use
`evaluator.close(wait=False)` to cancel it explicitly. Saving/hashing a snapshot,
IPC, and competing for CPU/GPU/memory bandwidth still cost time. Use a separate
GPU where useful and measure the effect.

The callback may return a numeric metric dictionary or `EvaluationResult` with
`metrics`, `samples` (prompt/output dictionaries), and `tables` (named row lists).
The combined callback report must be JSON-compatible and at most 1 MB; put large
trajectories in completed artifact files instead of returning them through IPC.
Put process startup under `if __name__ == "__main__"`. Callbacks must be top-level
importable functions, not lambdas or nested closures. These are local Python
callbacks in the user-owned training process, not uploaded server endpoints.

This helper evaluates held-out tasks. Training rollout generation and stale-policy
correction remain the responsibility of your PPO/GRPO/other RL implementation.

## Distributed loops

```python
from weave_support import Run
from weave_support.custom_training import distributed_context

with distributed_context(settings["devices"]) as context:
    with Run(rank=context.rank) as run:
        model = context.wrap(model)
        # Every rank executes the same update/command boundaries.
        commands = context.poll_commands(run)
        stop = run.handle_commands(checkpointer.save, commands=commands)
```

Launch with `torchrun --standalone --nproc_per_node=2 train.py`, or declare and
consume `strategy`/`devices` for a GUI DDP launch. The context initializes Gloo for
CPU or NCCL for CUDA, maps local ranks to the selected GPU indices, and destroys
only the process group it created. It does not spawn workers or seed RNGs.
`seed_everything(seed, rank_offset=context.rank)` is a separate optional helper.

Only rank zero reports. All ranks must participate in checkpoint saves and safe
command boundaries. This is DDP, not an RL rollout scheduler: it does not split
tasks, reduce/weight metrics, balance episode lengths, or guarantee equal backward
counts. Your sampler and algorithm must keep DDP collectives aligned. Per-rank RNG
and state restoration require the original world size.

When editing a resumed script, remember that `optimizer.load_state_dict()` restores
checkpoint hyperparameters too. Changing the optimizer constructor's learning rate
alone does not override the restored rate. Apply an intentional change after load,
and report the actual new configuration so the run record explains it.

Distributed ranks must share the run/checkpoint directory. These helpers do not
transfer files between machines or replace a sharded checkpoint system.
