"""Figures for the finished controlled He ablation.

Reads results/controlled_runs.json. Does not train and does not select checkpoints.
Confusion matrices are computed after that file already contains test metrics.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from main import confusion_matrix, load_mnist

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_JSON = PROJECT_DIR / "results" / "controlled_runs.json"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"
DIAGNOSTICS_PATH = PROJECT_DIR / "results" / "controlled_diagnostics.md"

VARIANT_ORDER = ("baseline", "l2", "batchnorm", "batchnorm_l2")


def load_payload(path=RUNS_JSON):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Refusing to invent a figure from an unfinished run.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("official_mnist"):
        raise RuntimeError("This file is not the official controlled MNIST result.")
    runs = payload["runs"]
    if len(runs) != 12 or any(not run.get("test_evaluated") for run in runs):
        raise RuntimeError("Test evaluation is incomplete. Not writing final figures.")
    return payload


def _runs_named(payload, name):
    return [run for run in payload["runs"] if run["variant"] == name]


def plot_paired_accuracy(payload, figure_dir):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    panels = ((0.0, "lambd = 0"), (0.04, "lambd = 0.04"))
    for axis, (lambd, title) in zip(axes, panels):
        for seed in (21, 7, 99):
            pair = [
                run for run in payload["runs"]
                if run["seed"] == seed and run["lambd"] == lambd
            ]
            pair = sorted(pair, key=lambda run: run["use_bnorm"])
            xs = [0, 1]
            ys = [run["test_accuracy"] for run in pair]
            axis.plot(xs, ys, marker="o", label=f"seed {seed}")
        axis.set_xticks([0, 1], ["no batch norm", "batch norm"])
        axis.set_title(title)
        axis.set_ylabel("test accuracy")
        axis.set_ylim(0.85, 1.0)
        axis.legend(frameon=False)
    figure.suptitle("Paired test accuracy, He initialization fixed")
    figure.tight_layout()
    figure.savefig(figure_dir / "paired_test_accuracy.png", dpi=140)
    plt.close(figure)


def plot_learning_curves(payload, figure_dir):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, name in zip(axes.ravel(), VARIANT_ORDER):
        for run in _runs_named(payload, name):
            epochs = range(1, run["epochs_ran"] + 1)
            axis.plot(epochs, run["history"]["val_cost"], label=f"seed {run['seed']}")
            best = run["best_epoch"] + 1
            axis.axvline(best, color="0.5", linewidth=0.8, linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation cost")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Validation cost. Dashed line is the restored checkpoint.")
    figure.tight_layout()
    figure.savefig(figure_dir / "val_learning_curves.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for run in payload["runs"]:
        epochs = range(1, run["epochs_ran"] + 1)
        axes[0].plot(epochs, run["history"]["val_cross_entropy"], color="C0", alpha=0.35)
        axes[1].plot(epochs, run["history"]["val_l2_penalty"], color="C1", alpha=0.45)
    axes[0].set_title("Validation cross-entropy")
    axes[1].set_title("Validation L2 term")
    for axis in axes:
        axis.set_xlabel("epoch")
    figure.suptitle("Components of the validation objective. L2 runs sit above zero on the right.")
    figure.tight_layout()
    figure.savefig(figure_dir / "val_cost_components.png", dpi=140)
    plt.close(figure)


def plot_best_epochs(payload, figure_dir):
    figure, axis = plt.subplots(figsize=(7, 4))
    for index, name in enumerate(VARIANT_ORDER):
        for run in _runs_named(payload, name):
            axis.scatter(index, run["best_epoch"], label=None)
            axis.text(index + 0.08, run["best_epoch"], str(run["seed"]), fontsize=8)
    axis.set_xticks(range(len(VARIANT_ORDER)), VARIANT_ORDER)
    axis.set_ylabel("best epoch index (from 0)")
    axis.set_title("Restored epoch for each of the 12 runs")
    figure.tight_layout()
    figure.savefig(figure_dir / "best_epochs.png", dpi=140)
    plt.close(figure)


def plot_weight_norms(payload, figure_dir):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    layers = ("W1", "W2", "W3", "W4")
    for axis, seed in zip(axes, (21, 7, 99)):
        seed_runs = [run for run in payload["runs"] if run["seed"] == seed]
        seed_runs = sorted(seed_runs, key=lambda run: VARIANT_ORDER.index(run["variant"]))
        for layer_index, layer in enumerate(layers):
            values = [run["parameter_norms"][layer] for run in seed_runs]
            positions = [index + layer_index * 0.15 for index in range(len(seed_runs))]
            axis.bar(positions, values, width=0.15, label=layer)
        axis.set_xticks(range(len(seed_runs)), [run["variant"] for run in seed_runs], rotation=20)
        axis.set_title(f"seed {seed}")
        axis.set_ylabel("L2 norm")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Weight norms at the restored checkpoint")
    figure.tight_layout()
    figure.savefig(figure_dir / "weight_norms.png", dpi=140)
    plt.close(figure)


def plot_confusion(payload, figure_dir):
    data = load_mnist(root=PROJECT_DIR / "data", val_fraction=0.2, seed=21)
    for seed in (21, 7, 99):
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        seed_runs = [run for run in payload["runs"] if run["seed"] == seed]
        seed_runs = sorted(seed_runs, key=lambda run: VARIANT_ORDER.index(run["variant"]))
        for axis, run in zip(axes, seed_runs):
            saved = torch.load(PROJECT_DIR / run["checkpoint"], map_location="cpu", weights_only=False)
            matrix = confusion_matrix(
                data["x_test"], data["y_test"], saved["parameters"], run["use_bnorm"]
            )
            axis.imshow(matrix, cmap="Blues")
            axis.set_title(run["variant"])
            axis.set_xlabel("predicted")
            axis.set_ylabel("true")
        figure.suptitle(f"Test confusion, seed {seed}. Not used to choose checkpoints.")
        figure.tight_layout()
        figure.savefig(figure_dir / f"confusion_seed{seed}.png", dpi=140)
        plt.close(figure)


def write_diagnostics(payload, path):
    lines = [
        "# Controlled-run diagnostics",
        "",
        "These notes describe the finished He ablation. They do not change the training objective.",
        "",
        "## L2 scaling",
        "",
        "The penalty in `cost_terms` is `lambd / (2m) * sum(W^2)`, where `m` is the number of",
        "columns in that call. The weight gradient adds `(lambd / m) * W`, which is the",
        "derivative of that penalty. Minibatch costs use `m` near 64. The validation cost",
        "uses `m = 12000`. The same weights therefore contribute a much larger penalty to the",
        "logged minibatch cost than to the validation cost. Early stopping still uses the sum,",
        "so this scale difference was left in place.",
        "",
        "Train penalty below is the mean minibatch penalty on the restored epoch. Validation",
        "penalty is the full validation matrix on that same checkpoint.",
        "",
        "| Variant | Seed | Train CE | Train L2 | Val CE | Val L2 | Train L2 / Val L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        best = run["best_epoch"]
        train_ce = run["history"]["train_cross_entropy"][best]
        train_l2 = run["history"]["train_l2_penalty"][best]
        val_ce = run["history"]["val_cross_entropy"][best]
        val_l2 = run["history"]["val_l2_penalty"][best]
        ratio = "n/a" if val_l2 == 0 else f"{train_l2 / val_l2:.1f}"
        lines.append(
            f"| {run['variant']} | {run['seed']} | {train_ce:.4f} | {train_l2:.4f} | "
            f"{val_ce:.4f} | {val_l2:.6f} | {ratio} |"
        )
    lines.extend([
        "",
        "Validation accuracy is stored beside validation cost in `controlled_runs.csv`.",
        "The two are not the same objective: a later epoch can have higher accuracy and worse cost,",
        "and the restored checkpoint is the cost minimum.",
        "",
        "## Batch norm",
        "",
        "The forward order is affine, ReLU, then batch norm on hidden layers. It was not moved",
        "before the ReLU. No separate affine-BN-ReLU ablation was run.",
        "Training with a batch of at least 2 uses the batch mean and variance and updates the",
        "running statistics. Evaluation, including accuracy, uses the running statistics.",
        "A training batch of size 1 falls back to those running statistics and does not update them.",
        "Gradient checks for batch norm on and off, with and without L2, are in",
        "`results/gradient_check_decoupled.json`.",
        "",
        "## Not run",
        "",
        "A lambda grid and a legacy-versus-He factorial were not run. The controlled comparison",
        "holds initialization fixed instead of estimating how much of the old gap was initialization.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    payload = load_payload()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_paired_accuracy(payload, FIGURE_DIR)
    plot_learning_curves(payload, FIGURE_DIR)
    plot_best_epochs(payload, FIGURE_DIR)
    plot_weight_norms(payload, FIGURE_DIR)
    plot_confusion(payload, FIGURE_DIR)
    write_diagnostics(payload, DIAGNOSTICS_PATH)
    print(f"wrote figures in {FIGURE_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
