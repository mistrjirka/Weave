"""CPU integration fixture exercising custom-loop reporting and exact restart."""

import time
import torch
from weave_support.model import Model
from weave_support.weave_training import Run
from weave_support.custom_training import Checkpointer, launch_settings

WEAVE_SUPPORTED_SETTINGS = ("devices",)
settings = launch_settings({"devices": [], "max_steps": 12})
torch.set_num_threads(1)
model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
with Run() as run:
    run.capabilities(
        metrics=True, commands=True, checkpoint=True, resume=True, samples=True
    )
    run.configure(
        {"devices": [], "precision": "fp32", "strategy": "single"},
        metrics={
            "eval/success": {
                "label": "Task success",
                "direction": "max",
                "min": 0,
                "max": 1,
            },
            "train/reward": {"label": "Mean reward"},
        },
        primary_metric="eval/success",
        headline_metrics=["eval/success", "train/reward"],
        step_unit="policy updates",
    )
    checkpoint = Checkpointer(
        run, components={"model": model, "optimizer": optimizer}, exact_resume=True
    )
    if settings.get("checkpoint"):
        checkpoint.load(
            settings["checkpoint"], weights_only=settings.get("start_mode") != "resume"
        )
    while run.step < settings["max_steps"]:
        optimizer.zero_grad()
        batch = torch.randint(0, 8, (2, 8))
        loss = model(batch).square().mean()
        loss.backward()
        optimizer.step()
        step = run.step + 1
        run.log(
            {
                "train/loss": loss.detach(),
                "train/reward": 0.2 + step / 100,
                "eval/success": step / settings["max_steps"],
            },
            step=step,
            counters={"episodes": step * 4, "rollouts": step},
        )
        run.progress(completed=step, total=settings["max_steps"], unit="policy updates")
        if step == 1:
            run.table(
                "Trajectories",
                [
                    {
                        "task": "Example task",
                        "call": '<call name="demo"/>',
                        "reward": 1.0,
                        "success": True,
                    }
                ],
                metadata={"reward_version": "fixture-v1"},
            )
            run.sample(prompt="Example task", output='<call name="demo"/>')
        time.sleep(0.08)
        if run.handle_commands(checkpoint.save):
            break
    else:
        checkpoint.save()
