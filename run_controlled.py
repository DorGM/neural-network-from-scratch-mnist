"""Controlled He-initialization ablation on official MNIST.

Every variant uses initialization='he'. Batch norm does not choose the scheme.
Legacy checkpoints in checkpoints/*.pt are never read or written.

Test predictions are computed only after a selection lock without test metrics
has been written. python run_controlled.py --smoke-test uses synthetic data
and is not an MNIST result.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import torch
import torchvision

from main import (
    DEFAULT_LAYER_DIMS,
    PIXEL_SCALE,
    load_mnist,
    predict,
    seeded_parameters,
    train_model,
    weight_norms,
)

EXPERIMENT_ID = "controlled_he_v1"
INITIALIZATION = "he"
SPLIT_SEED = 21
SEEDS = (21, 7, 99)
LEARNING_RATE = 0.009
MAX_EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 20
VAL_FRACTION = 0.2
VARIANTS = (
    {"name": "baseline", "use_bnorm": False, "lambd": 0.0},
    {"name": "l2", "use_bnorm": False, "lambd": 0.04},
    {"name": "batchnorm", "use_bnorm": True, "lambd": 0.0},
    {"name": "batchnorm_l2", "use_bnorm": True, "lambd": 0.04},
)

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / EXPERIMENT_ID
LEGACY_CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
SELECTION_LOCK = RESULTS_DIR / "controlled_he_v1_selection_lock.json"
RUNS_CSV = RESULTS_DIR / "controlled_runs.csv"
RUNS_JSON = RESULTS_DIR / "controlled_runs.json"
SUMMARY_PATH = RESULTS_DIR / "controlled_summary.md"
MANIFEST_PATH = RESULTS_DIR / "experiment_manifest.json"

TEST_EXPOSURE = (
    "The learning rate, lambda 0.04, architecture, split, batch size, and "
    "patience match the already published legacy experiment, so those choices "
    "were not blind to the MNIST test set. Checkpoints in this controlled "
    "experiment were selected on validation cost before any test prediction "
    "for these runs was computed."
)


def code_hash():
    # Line endings are not a code change. OneDrive may rewrite LF as CRLF.
    source = (PROJECT_DIR / "main.py").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(source).hexdigest()


def package_versions():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "num_threads": int(torch.get_num_threads()),
    }


def run_config(variant, seed):
    if INITIALIZATION != "he":
        raise RuntimeError("The controlled experiment must select He explicitly.")
    return {
        "experiment_id": EXPERIMENT_ID,
        "variant": variant["name"],
        "use_bnorm": bool(variant["use_bnorm"]),
        "lambd": float(variant["lambd"]),
        "initialization": INITIALIZATION,
        "seed": int(seed),
        "seed_meaning": "initialization and minibatch shuffle; the split is not redrawn",
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "layer_dims": list(DEFAULT_LAYER_DIMS),
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "pixel_scale": PIXEL_SCALE,
        "selection_metric": "val_cost",
        "device": "cpu",
        "main_sha256": code_hash(),
    }


def signature_for(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_path(variant_name, seed, signature):
    if CHECKPOINT_DIR == LEGACY_CHECKPOINT_DIR:
        raise RuntimeError("Controlled checkpoints must not use the legacy directory.")
    filename = f"{EXPERIMENT_ID}_{variant_name}_seed{seed}_{signature[:12]}.pt"
    path = CHECKPOINT_DIR / filename
    legacy_path = LEGACY_CHECKPOINT_DIR / f"{variant_name}_seed{seed}.pt"
    if path.resolve() == legacy_path.resolve():
        raise RuntimeError(f"Refusing to touch legacy checkpoint {legacy_path.name}.")
    return path


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def assert_shared_initialization(seed):
    plain = seeded_parameters(DEFAULT_LAYER_DIMS, False, INITIALIZATION, seed, "cpu")
    normalized = seeded_parameters(DEFAULT_LAYER_DIMS, True, INITIALIZATION, seed, "cpu")
    for layer in range(1, len(DEFAULT_LAYER_DIMS)):
        if not torch.equal(plain[f"W{layer}"], normalized[f"W{layer}"]):
            raise RuntimeError(f"Seed {seed} does not share W{layer} across batch norm.")
        if not torch.equal(plain[f"b{layer}"], normalized[f"b{layer}"]):
            raise RuntimeError(f"Seed {seed} does not share b{layer} across batch norm.")
    if "gamma1" not in normalized or "gamma1" in plain:
        raise RuntimeError("Batch-norm scale and shift were not kept separate.")


def _best_index(history):
    costs = history["val_cost"]
    return min(range(len(costs)), key=costs.__getitem__)


def _row_from_checkpoint(saved, path):
    history = saved["history"]
    best = history["best_epoch"]
    if best != _best_index(history):
        raise RuntimeError(f"{path.name} best_epoch is not the first minimum validation cost.")
    if history.get("initialization") != INITIALIZATION:
        raise RuntimeError(f"{path.name} was not trained with He initialization.")
    if history.get("selection_metric") != "val_cost":
        raise RuntimeError(f"{path.name} did not record validation-cost selection.")
    norms = weight_norms(saved["parameters"])
    config = saved["config"]
    return {
        "experiment_id": EXPERIMENT_ID,
        "variant": config["variant"],
        "use_bnorm": config["use_bnorm"],
        "lambd": config["lambd"],
        "initialization": config["initialization"],
        "seed": config["seed"],
        "split_seed": config["split_seed"],
        "best_epoch": best,
        "epochs_ran": history["epochs_ran"],
        "best_val_loss": history["val_cost"][best],
        "best_val_accuracy": history["val_accuracy"][best],
        "best_val_cross_entropy": history["val_cross_entropy"][best],
        "best_val_l2_penalty": history["val_l2_penalty"][best],
        "best_train_accuracy": history["train_accuracy"][best],
        "best_train_cross_entropy": history["train_cross_entropy"][best],
        "best_train_l2_penalty": history["train_l2_penalty"][best],
        "parameter_norms": norms,
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "max_epochs": config["max_epochs"],
        "patience": config["patience"],
        "layer_dims": config["layer_dims"],
        "device": "cpu",
        "versions": saved["versions"],
        "main_sha256": config["main_sha256"],
        "signature": saved["signature"],
        "checkpoint": path.relative_to(PROJECT_DIR).as_posix(),
        "status": "selection_locked",
        "test_evaluated": False,
        "history": {
            "train_cost": history["train_cost"],
            "val_cost": history["val_cost"],
            "train_accuracy": history["train_accuracy"],
            "val_accuracy": history["val_accuracy"],
            "train_cross_entropy": history["train_cross_entropy"],
            "train_l2_penalty": history["train_l2_penalty"],
            "val_cross_entropy": history["val_cross_entropy"],
            "val_l2_penalty": history["val_l2_penalty"],
        },
    }


def _manifest_entry(row):
    return {
        "experiment_id": row["experiment_id"],
        "variant": row["variant"],
        "seed": row["seed"],
        "use_bnorm": row["use_bnorm"],
        "lambd": row["lambd"],
        "initialization": row["initialization"],
        "status": row["status"],
        "checkpoint": row.get("checkpoint"),
        "signature": row.get("signature"),
        "best_epoch": row.get("best_epoch"),
        "epochs_ran": row.get("epochs_ran"),
        "best_val_loss": row.get("best_val_loss"),
        "test_evaluated": row.get("test_evaluated", False),
        "official_mnist": True,
    }


def write_manifest(rows, note):
    existing = {}
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    others = [
        item for item in existing.get("runs", [])
        if item.get("experiment_id") != EXPERIMENT_ID
    ]
    existing["runs"] = others + [_manifest_entry(row) for row in rows]
    existing["controlled_experiment_id"] = EXPERIMENT_ID
    existing["note"] = note
    atomic_write_text(MANIFEST_PATH, json.dumps(existing, indent=2) + "\n")


def _load_valid_checkpoint(path, config, signature):
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("signature") != signature or saved.get("config") != config:
        print(f"refusing to reuse {path.name}: config or signature differs", flush=True)
        return None
    if saved.get("test_metrics") not in (None, {}):
        print(f"refusing to reuse {path.name}: file already holds test metrics", flush=True)
        return None
    return saved


def _planned_rows():
    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            config = run_config(variant, seed)
            signature = signature_for(config)
            path = checkpoint_path(variant["name"], seed, signature)
            rows.append({
                "experiment_id": EXPERIMENT_ID,
                "variant": variant["name"],
                "use_bnorm": variant["use_bnorm"],
                "lambd": variant["lambd"],
                "initialization": INITIALIZATION,
                "seed": seed,
                "status": "not_started",
                "test_evaluated": False,
                "checkpoint": path.relative_to(PROJECT_DIR).as_posix(),
                "signature": signature,
                "config": config,
            })
    return rows


def train_until_selection_locked(data):
    if data["x_train"].shape[1] != 48000 or data["y_train"].shape[1] != 48000:
        raise RuntimeError("Fit split is not 48,000 columns.")
    if data["x_val"].shape[1] != 12000 or data["y_val"].shape[1] != 12000:
        raise RuntimeError("Validation split is not 12,000 columns.")
    planned = _planned_rows()
    order = [(row["variant"], row["seed"]) for row in planned]
    state = {(row["variant"], row["seed"]): row for row in planned}
    write_manifest([state[key] for key in order], "controlled run started; test set not evaluated")
    versions = package_versions()
    for seed in SEEDS:
        assert_shared_initialization(seed)
        for variant in VARIANTS:
            config = run_config(variant, seed)
            signature = signature_for(config)
            path = checkpoint_path(variant["name"], seed, signature)
            try:
                saved = _load_valid_checkpoint(path, config, signature)
                if saved is None:
                    print(
                        f"training {variant['name']} seed {seed} initialization {INITIALIZATION}",
                        flush=True,
                    )
                    parameters, history = train_model(
                        data["x_train"],
                        data["y_train"],
                        data["x_val"],
                        data["y_val"],
                        DEFAULT_LAYER_DIMS,
                        learning_rate=LEARNING_RATE,
                        num_iterations=MAX_EPOCHS,
                        batch_size=BATCH_SIZE,
                        use_bnorm=variant["use_bnorm"],
                        lambd=variant["lambd"],
                        initialization=INITIALIZATION,
                        patience=PATIENCE,
                        seed=seed,
                        device="cpu",
                        log_every=5,
                    )
                    parameters = {name: value.detach().cpu() for name, value in parameters.items()}
                    saved = {
                        "parameters": parameters,
                        "history": history,
                        "config": config,
                        "signature": signature,
                        "versions": versions,
                        "test_metrics": None,
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(saved, path)
                    reloaded = _load_valid_checkpoint(path, config, signature)
                    if reloaded is None:
                        raise RuntimeError(f"Saved checkpoint failed its own signature check: {path.name}")
                    saved = reloaded
                else:
                    print(f"loaded {path.name}", flush=True)
                row = _row_from_checkpoint(saved, path)
                val_metrics = predict(
                    data["x_val"], data["y_val"], saved["parameters"], variant["use_bnorm"]
                )
                if abs(val_metrics["accuracy"] - row["best_val_accuracy"]) > 1e-6:
                    raise RuntimeError(f"Restored validation accuracy mismatch for {path.name}")
                state[(variant["name"], seed)] = row
                print(
                    f"selected {variant['name']} seed {seed} epochs {row['epochs_ran']} "
                    f"best {row['best_epoch']} val_loss {row['best_val_loss']:.6f} "
                    f"val_acc {row['best_val_accuracy']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                state[(variant["name"], seed)]["status"] = "failed"
                state[(variant["name"], seed)]["error"] = str(exc)
                state[(variant["name"], seed)]["test_evaluated"] = False
                write_manifest([state[key] for key in order], "controlled run failed before test evaluation")
                raise
            write_manifest([state[key] for key in order], "selection in progress; test set not evaluated")
    finished = [state[key] for key in order]
    if any(row["status"] != "selection_locked" for row in finished):
        raise RuntimeError("Not every controlled run reached a validation checkpoint.")
    return finished


def write_selection_lock(rows):
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "selection_metric": "val_cost",
        "test_metrics_present": False,
        "initialization": INITIALIZATION,
        "main_sha256": code_hash(),
        "runs": [
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "use_bnorm": row["use_bnorm"],
                "lambd": row["lambd"],
                "signature": row["signature"],
                "checkpoint": row["checkpoint"],
                "best_epoch": row["best_epoch"],
                "epochs_ran": row["epochs_ran"],
                "best_val_loss": row["best_val_loss"],
                "best_val_accuracy": row["best_val_accuracy"],
            }
            for row in rows
        ],
    }
    text = json.dumps(payload, indent=2) + "\n"
    if "test_accuracy" in text:
        raise RuntimeError("Selection lock must not contain test accuracy.")
    atomic_write_text(SELECTION_LOCK, text)
    reread = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    if reread["test_metrics_present"] is not False or len(reread["runs"]) != 12:
        raise RuntimeError("Selection lock on disk is incomplete.")
    for run in reread["runs"]:
        if "test_accuracy" in run:
            raise RuntimeError("Selection lock run contains test accuracy.")
    return hashlib.sha256(SELECTION_LOCK.read_bytes()).hexdigest()


def evaluate_test(rows, x_test, y_test, lock_hash):
    if hashlib.sha256(SELECTION_LOCK.read_bytes()).hexdigest() != lock_hash:
        raise RuntimeError("Selection lock changed before test evaluation.")
    if x_test.shape[1] != 10000 or y_test.shape[1] != 10000:
        raise RuntimeError("Official test split is not 10,000 columns.")
    for row in rows:
        path = PROJECT_DIR / row["checkpoint"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("test_metrics") not in (None, {}):
            raise RuntimeError(f"{path.name} already contains test metrics.")
        metrics = predict(x_test, y_test, saved["parameters"], row["use_bnorm"])
        row["test_accuracy"] = metrics["accuracy"]
        row["test_correct"] = metrics["correct"]
        row["test_total"] = metrics["total"]
        row["status"] = "completed"
        row["test_evaluated"] = True
        print(
            f"test {row['variant']} seed {row['seed']} "
            f"{metrics['correct']}/{metrics['total']}",
            flush=True,
        )
    return rows


def _mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def _find_row(rows, seed, use_bnorm, lambd):
    matches = [
        row for row in rows
        if row["seed"] == seed and row["use_bnorm"] is use_bnorm and row["lambd"] == lambd
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for seed {seed}, bn {use_bnorm}, lambd {lambd}.")
    return matches[0]


def paired_differences(rows):
    report = {}
    for lambd in (0.0, 0.04):
        per_seed = []
        for seed in SEEDS:
            batch_norm = _find_row(rows, seed, True, lambd)
            baseline = _find_row(rows, seed, False, lambd)
            per_seed.append({
                "seed": seed,
                "test_accuracy": batch_norm["test_accuracy"] - baseline["test_accuracy"],
                "val_accuracy": batch_norm["best_val_accuracy"] - baseline["best_val_accuracy"],
                "val_loss": batch_norm["best_val_loss"] - baseline["best_val_loss"],
            })
        summary = {}
        for metric in ("test_accuracy", "val_accuracy", "val_loss"):
            mean, sample_sd = _mean_sd([item[metric] for item in per_seed])
            summary[metric] = {"mean": mean, "sample_sd": sample_sd}
        report[str(lambd)] = {"per_seed": per_seed, "mean_and_sample_sd": summary}
    return report


def variant_aggregates(rows):
    aggregates = []
    for variant in VARIANTS:
        group = [row for row in rows if row["variant"] == variant["name"]]
        test_mean, test_sd = _mean_sd([row["test_accuracy"] for row in group])
        val_mean, val_sd = _mean_sd([row["best_val_accuracy"] for row in group])
        loss_mean, loss_sd = _mean_sd([row["best_val_loss"] for row in group])
        aggregates.append({
            "variant": variant["name"],
            "test_mean": test_mean,
            "test_sample_sd": test_sd,
            "val_accuracy_mean": val_mean,
            "val_accuracy_sample_sd": val_sd,
            "val_loss_mean": loss_mean,
            "val_loss_sample_sd": loss_sd,
        })
    return aggregates


def _fmt(value, digits):
    return f"{value:.{digits}f}"


def write_summary(rows, paired, aggregates, lock_hash):
    lines = [
        "# Controlled MNIST ablation (He initialization fixed)",
        "",
        "This is the primary comparison. Every variant uses `initialization=\"he\"`.",
        "Batch norm does not choose the initializer. The legacy confounded numbers",
        "remain in `results/legacy_mnist_summary.md`.",
        "",
        "## Seeds",
        "",
        "Split seed 21 draws the 48,000 / 12,000 partition once and is shared by every run.",
        "Seeds 21, 7, and 99 each seed weight initialization and the minibatch permutation.",
        "They do not redraw the split. The same seed produces the same initial W and b",
        "for the batch-norm and non-batch-norm variants. Gamma and beta start at ones and zeros.",
        "",
        "## Selection",
        "",
        "Each checkpoint is the first minimum of validation cost (cross-entropy plus the",
        "L2 term when `lambd > 0`). The L2 term uses `lambd / (2m) * ||W||^2` with `m`",
        "equal to the column count of that call, so a minibatch and the full validation",
        "set do not apply the same penalty to the same weights. That scaling was not changed.",
        "",
        TEST_EXPOSURE,
        "",
        f"Selection lock SHA-256: `{lock_hash}`.",
        "",
        "## Per-seed results",
        "",
        "| Variant | Seed | Epochs | Best epoch | Val loss | Val accuracy | Test accuracy | Test correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['epochs_ran']} | {row['best_epoch']} | "
            f"{_fmt(row['best_val_loss'], 6)} | {_fmt(row['best_val_accuracy'], 4)} | "
            f"{_fmt(row['test_accuracy'], 4)} | {row['test_correct']}/{row['test_total']} |"
        )
    lines.extend([
        "",
        "Means and sample standard deviations use divisor n-1 over the three seeds.",
        "",
        "| Variant | Val loss mean | Val loss sample SD | Val acc mean | Val acc sample SD | Test mean | Test sample SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in aggregates:
        lines.append(
            f"| {item['variant']} | {_fmt(item['val_loss_mean'], 6)} | {_fmt(item['val_loss_sample_sd'], 6)} | "
            f"{_fmt(item['val_accuracy_mean'], 4)} | {_fmt(item['val_accuracy_sample_sd'], 4)} | "
            f"{_fmt(item['test_mean'], 4)} | {_fmt(item['test_sample_sd'], 4)} |"
        )
    lines.extend([
        "",
        "## Paired differences",
        "",
        "Difference = batch norm minus no batch norm, at the same lambda and seed.",
        "A negative validation-loss difference means the batch-norm run has the lower validation cost.",
        "Three seeds are not a significance test.",
        "",
    ])
    for lambd, block in paired.items():
        lines.append(f"### lambd = {lambd}")
        lines.append("")
        lines.append("| Seed | Test accuracy | Val accuracy | Val loss |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for item in block["per_seed"]:
            lines.append(
                f"| {item['seed']} | {_fmt(item['test_accuracy'], 4)} | "
                f"{_fmt(item['val_accuracy'], 4)} | {_fmt(item['val_loss'], 6)} |"
            )
        summary = block["mean_and_sample_sd"]
        lines.append("")
        lines.append(
            f"Mean test-accuracy difference {_fmt(summary['test_accuracy']['mean'], 4)}, "
            f"sample SD {_fmt(summary['test_accuracy']['sample_sd'], 4)}."
        )
        lines.append(
            f"Mean validation-accuracy difference {_fmt(summary['val_accuracy']['mean'], 4)}, "
            f"sample SD {_fmt(summary['val_accuracy']['sample_sd'], 4)}."
        )
        lines.append(
            f"Mean validation-loss difference {_fmt(summary['val_loss']['mean'], 6)}, "
            f"sample SD {_fmt(summary['val_loss']['sample_sd'], 6)}."
        )
        lines.append("")
    lines.extend([
        "## What this sample can support",
        "",
        "The table is the result for this fixed architecture, this one split, and these three seeds.",
        "It does not by itself establish a general advantage for batch norm or for L2.",
        "",
    ])
    text = "\n".join(lines)
    atomic_write_text(SUMMARY_PATH, text)
    return text


def write_result_files(rows, lock_hash):
    if any(not row["test_evaluated"] for row in rows):
        raise RuntimeError("Refusing to write final results before every test evaluation.")
    paired = paired_differences(rows)
    aggregates = variant_aggregates(rows)
    versions = rows[0]["versions"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "official_mnist": True,
        "initialization": INITIALIZATION,
        "initialization_coupled_to_batch_norm": False,
        "test_exposure": TEST_EXPOSURE,
        "selection_lock_sha256": lock_hash,
        "selection_metric": "val_cost",
        "versions": versions,
        "main_sha256": code_hash(),
        "runs": rows,
        "variant_aggregates": aggregates,
        "paired_differences": paired,
    }
    atomic_write_text(RUNS_JSON, json.dumps(payload, indent=2) + "\n")
    fieldnames = [
        "experiment_id", "variant", "use_bnorm", "lambd", "initialization", "seed",
        "split_seed", "best_epoch", "epochs_ran", "best_val_loss", "best_val_accuracy",
        "best_val_cross_entropy", "best_val_l2_penalty", "best_train_accuracy",
        "best_train_cross_entropy", "best_train_l2_penalty", "test_accuracy",
        "test_correct", "test_total", "W1", "W2", "W3", "W4", "learning_rate",
        "batch_size", "max_epochs", "patience", "device", "torch", "torchvision",
        "main_sha256", "signature", "checkpoint", "status",
    ]
    temporary = RUNS_CSV.with_suffix(".csv.tmp")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "experiment_id": row["experiment_id"],
                "variant": row["variant"],
                "use_bnorm": row["use_bnorm"],
                "lambd": row["lambd"],
                "initialization": row["initialization"],
                "seed": row["seed"],
                "split_seed": row["split_seed"],
                "best_epoch": row["best_epoch"],
                "epochs_ran": row["epochs_ran"],
                "best_val_loss": row["best_val_loss"],
                "best_val_accuracy": row["best_val_accuracy"],
                "best_val_cross_entropy": row["best_val_cross_entropy"],
                "best_val_l2_penalty": row["best_val_l2_penalty"],
                "best_train_accuracy": row["best_train_accuracy"],
                "best_train_cross_entropy": row["best_train_cross_entropy"],
                "best_train_l2_penalty": row["best_train_l2_penalty"],
                "test_accuracy": row["test_accuracy"],
                "test_correct": row["test_correct"],
                "test_total": row["test_total"],
                "W1": row["parameter_norms"]["W1"],
                "W2": row["parameter_norms"]["W2"],
                "W3": row["parameter_norms"]["W3"],
                "W4": row["parameter_norms"]["W4"],
                "learning_rate": row["learning_rate"],
                "batch_size": row["batch_size"],
                "max_epochs": row["max_epochs"],
                "patience": row["patience"],
                "device": row["device"],
                "torch": row["versions"]["torch"],
                "torchvision": row["versions"]["torchvision"],
                "main_sha256": row["main_sha256"],
                "signature": row["signature"],
                "checkpoint": row["checkpoint"],
                "status": row["status"],
            })
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(RUNS_CSV)
    write_summary(rows, paired, aggregates, lock_hash)
    write_manifest(rows, "controlled run completed after validation selection and then test evaluation")
    return payload


def run_controlled():
    for seed in SEEDS:
        assert_shared_initialization(seed)
    print("loading MNIST from the local data directory", flush=True)
    data = load_mnist(root=PROJECT_DIR / "data", val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    rows = train_until_selection_locked(data)
    print("writing selection lock before test evaluation", flush=True)
    lock_hash = write_selection_lock(rows)
    print(f"selection lock {lock_hash}", flush=True)
    evaluate_test(rows, data["x_test"], data["y_test"], lock_hash)
    write_result_files(rows, lock_hash)
    print(f"wrote {RUNS_CSV.name} and {SUMMARY_PATH.name}", flush=True)


def run_smoke(output_dir):
    """Tiny synthetic run. official_mnist is false. This does not load MNIST."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    features, classes, n_train, n_val = 6, 3, 40, 12
    x_train = torch.rand(features, n_train, generator=generator)
    x_val = torch.rand(features, n_val, generator=generator)
    train_labels = torch.randint(0, classes, (n_train,), generator=generator)
    val_labels = torch.randint(0, classes, (n_val,), generator=generator)
    y_train = torch.zeros(classes, n_train)
    y_val = torch.zeros(classes, n_val)
    y_train[train_labels, torch.arange(n_train)] = 1.0
    y_val[val_labels, torch.arange(n_val)] = 1.0
    _parameters, history = train_model(
        x_train,
        y_train,
        x_val,
        y_val,
        [features, 4, classes],
        learning_rate=0.05,
        num_iterations=3,
        batch_size=8,
        use_bnorm=True,
        lambd=0.01,
        initialization="he",
        patience=2,
        seed=0,
        device="cpu",
    )
    summary = {
        "official_mnist": False,
        "label": "synthetic smoke test",
        "initialization": history["initialization"],
        "use_bnorm": history["use_bnorm"],
        "epochs_ran": history["epochs_ran"],
        "best_epoch": history["best_epoch"],
        "finite_val_cost": math_isfinite(history["val_cost"][history["best_epoch"]]),
    }
    atomic_write_text(output_dir / "smoke_test.json", json.dumps(summary, indent=2) + "\n")
    return summary


def math_isfinite(value):
    return value == value and value not in (float("inf"), float("-inf"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled He MNIST ablation or a synthetic smoke test.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train a few steps on synthetic data. Not an official MNIST run.",
    )
    args = parser.parse_args(argv)
    if args.smoke_test:
        summary = run_smoke(RESULTS_DIR)
        print(json.dumps(summary), flush=True)
        return 0
    run_controlled()
    return 0


if __name__ == "__main__":
    sys.exit(main())
