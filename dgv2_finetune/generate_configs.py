#!/usr/bin/env python3
"""Generate the preregistered SUSTech1K-to-CASIA-B fine-tuning configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
# The repository holds two CASIA-B split protocols that differ only in subject
# 075 (custom: train pool 001-075 / test 076-124 = 49 subjects; OpenGait
# standard: test 075-124 = 50 subjects). Mixing them raises no error but makes
# accuracies incomparable, so every config's partition must match this
# canonical 49-subject holdout.
CANONICAL_PARTITION = REPO_ROOT / "datasets/CASIA-B/CASIA-B.json"
CANONICAL_TEST_SET = json.loads(CANONICAL_PARTITION.read_text())["TEST_SET"]
SOURCE_DIR = REPO_ROOT / "configs/deepgaitv2"
OUTPUT_DIR = SOURCE_DIR / "finetune_sustech"
WARM_START = (
    "./output/SUSTech1K/DeepGaitV2/DeepGaitV2/checkpoints/"
    "DeepGaitV2_30_DA-50000_no_fc_bin.pt"
)

SUBSET_SOURCES = {
    "default": "DeepGaitV2_casiab-nAG-ryu-default.yaml",
    "nm2": "DeepGaitV2_casiab-nAG-ryu-nm2.yaml",
    "bg2": "DeepGaitV2_casiab-nAG-ryu-bg2.yaml",
    "cl2": "DeepGaitV2_casiab-nAG-ryu-cl2.yaml",
    "nm1-bg1": "DeepGaitV2_casiab-nAG-ryu-nm1-bg1.yaml",
    "nm1-cl1": "DeepGaitV2_casiab-nAG-ryu-nm1-cl1.yaml",
    "bg1-cl1": "DeepGaitV2_casiab-nAG-bg1-cl1.yaml",
    "000-180": "DeepGaitV2_casiab-nAG-000-180.yaml",
    "000-090": "DeepGaitV2_casiab-nAG-ryu-000-090.yaml",
    "090-180": "DeepGaitV2_casiab-nAG-090-180.yaml",
    "nm1-bg1-cl1": "DeepGaitV2_casiab-nAG-nm1-bg1-cl1.yaml",
    "nm2-bg2-cl2": "DeepGaitV2_casiab-nAG-ryu-nm2bg2cl2.yaml",
    "nm6": "DeepGaitV2_casiab-nAG-nm6.yaml",
}
SEEDS = {subset: (0,) for subset in SUBSET_SOURCES}
SEEDS["000-180"] = (0, 1)
SEEDS["nm1-cl1"] = (0, 1)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return config


def validate_source(config: Dict[str, Any], path: Path) -> None:
    if config["model_cfg"]["SeparateBNNecks"]["class_num"] != 75:
        raise ValueError(f"class_num must be 75 in {path}")
    partition = REPO_ROOT / config["data_cfg"]["dataset_partition"]
    if json.loads(partition.read_text())["TEST_SET"] != CANONICAL_TEST_SET:
        raise ValueError(
            f"TEST_SET mismatch in {partition} (used by {path}): expected the "
            f"shared 49-subject holdout 076-124 from {CANONICAL_PARTITION.name}. "
            "Standard-protocol partitions such as CASIA-B-20-2.json test on 50 "
            "subjects including 075 and are silently incomparable."
        )
    sampler = config["trainer_cfg"]["sampler"]
    if sampler["batch_size"] != [8, 16] or sampler["frames_num_fixed"] != 30:
        raise ValueError(f"Unexpected training sampler in {path}")

    transform = config["trainer_cfg"]["transform"][0]
    transforms = transform.get("trf_cfg", [])
    augmentation_probabilities = {
        item["type"]: item.get("prob")
        for item in transforms
        if item["type"] in {"RandomPerspective", "RandomHorizontalFlip", "RandomRotate"}
    }
    expected = {
        "RandomPerspective": 0.0,
        "RandomHorizontalFlip": 0.0,
        "RandomRotate": 0.0,
    }
    if augmentation_probabilities != expected:
        raise ValueError(f"Augmentation must be disabled in {path}")


def build_config(source: Path, subset: str, seed: int) -> Dict[str, Any]:
    config = load_yaml(source)
    validate_source(config, source)

    save_name = f"DeepGaitV2-ft-sustech-{subset}-s{seed}"
    config["base_seed"] = seed
    config["optimizer_cfg"]["lr"] = 0.01
    config["scheduler_cfg"]["milestones"] = [10000, 20000, 25000]

    trainer = config["trainer_cfg"]
    trainer.update(
        {
            "restore_ckpt_strict": False,
            "restore_hint": WARM_START,
            "optimizer_reset": True,
            "scheduler_reset": True,
            "save_iter": 5000,
            "save_name": save_name,
            "total_iter": 30000,
        }
    )

    evaluator = config["evaluator_cfg"]
    evaluator.update(
        {
            "restore_ckpt_strict": True,
            "restore_hint": 30000,
            "save_name": save_name,
        }
    )
    return config


def render(config: Dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed configs instead of writing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stale = []
    expected_paths = set()
    if not args.check:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for subset, source_name in SUBSET_SOURCES.items():
        source = SOURCE_DIR / source_name
        for seed in SEEDS[subset]:
            output = OUTPUT_DIR / f"DeepGaitV2_casiab-ft-sustech-{subset}-s{seed}.yaml"
            expected_paths.add(output)
            content = render(build_config(source, subset, seed))
            if args.check:
                if not output.is_file() or output.read_text() != content:
                    stale.append(output)
            else:
                output.write_text(content)
                print(output.relative_to(REPO_ROOT))

    unexpected = sorted(OUTPUT_DIR.glob("*.yaml")) if OUTPUT_DIR.exists() else []
    unexpected = [path for path in unexpected if path not in expected_paths]
    if unexpected:
        raise RuntimeError(
            "Unexpected fine-tuning configs: "
            + ", ".join(str(path.relative_to(REPO_ROOT)) for path in unexpected)
        )
    if stale:
        raise RuntimeError(
            "Generated configs are missing or stale: "
            + ", ".join(str(path.relative_to(REPO_ROOT)) for path in stale)
        )
    if args.check:
        print(f"Validated {len(expected_paths)} generated configs")


if __name__ == "__main__":
    main()
