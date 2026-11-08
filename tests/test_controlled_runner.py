"""Runner checks that do not download MNIST or start the 12 official runs."""

import json

import run_controlled


def test_controlled_config_fixes_he_and_avoids_legacy_checkpoints():
    assert run_controlled.INITIALIZATION == "he"
    assert run_controlled.CHECKPOINT_DIR != run_controlled.LEGACY_CHECKPOINT_DIR
    signatures = set()
    for seed in run_controlled.SEEDS:
        for variant in run_controlled.VARIANTS:
            config = run_controlled.run_config(variant, seed)
            assert config["initialization"] == "he"
            assert config["split_seed"] == 21
            assert config["max_epochs"] == 300
            assert config["patience"] == 20
            assert config["learning_rate"] == 0.009
            assert config["batch_size"] == 64
            signature = run_controlled.signature_for(config)
            path = run_controlled.checkpoint_path(variant["name"], seed, signature)
            assert run_controlled.EXPERIMENT_ID in path.name
            assert variant["name"] in path.name
            assert f"seed{seed}" in path.name
            assert path.parent == run_controlled.CHECKPOINT_DIR
            signatures.add(signature)
    assert len(signatures) == 12
    changed = dict(run_controlled.run_config(run_controlled.VARIANTS[0], 21))
    changed["lambd"] = 0.2
    assert run_controlled.signature_for(changed) not in signatures


def test_smoke_test_is_not_an_mnist_run(tmp_path):
    summary = run_controlled.run_smoke(tmp_path)
    assert summary["official_mnist"] is False
    assert summary["initialization"] == "he"
    assert summary["finite_val_cost"] is True
    written = json.loads((tmp_path / "smoke_test.json").read_text(encoding="utf-8"))
    assert written["official_mnist"] is False
    assert not (tmp_path / "controlled_runs.csv").exists()
