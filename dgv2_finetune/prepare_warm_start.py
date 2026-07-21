#!/usr/bin/env python3
"""Create the fixed SUSTech1K warm-start checkpoint for CASIA-B fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "output/SUSTech1K/DeepGaitV2/DeepGaitV2/checkpoints"
    / "DeepGaitV2_30_DA-50000.pt"
)
DERIVED_CHECKPOINT = SOURCE_CHECKPOINT.with_name(
    "DeepGaitV2_30_DA-50000_no_fc_bin.pt"
)
SOURCE_SHA256 = "da3517cc514af5ae17d3e489511838e7c081b9091d810eb07d7ae6473d48d9fe"
REMOVED_KEY = "BNNecks.fc_bin"
REMOVED_TRAINING_STATE_KEYS = ("optimizer", "scheduler")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    # weights_only was added after the PyTorch version used by some OpenGait
    # environments. The fixed local checkpoint is trusted research input.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed SUSTech1K warm-start checkpoint by removing "
            f"{REMOVED_KEY} and all optimizer/scheduler state. Existing output "
            "is never overwritten."
        )
    )
    parser.add_argument("--source", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DERIVED_CHECKPOINT)
    parser.add_argument(
        "--expected-source-sha256",
        default=SOURCE_SHA256,
        help="refuse to run if the source hash differs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if source == output:
        raise ValueError("Source and output checkpoint paths must differ")
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_hash_before = sha256(source)
    if source_hash_before != args.expected_source_sha256:
        raise ValueError(
            "Unexpected source checkpoint hash: "
            f"expected {args.expected_source_sha256}, got {source_hash_before}"
        )

    checkpoint = load_checkpoint(source)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise TypeError("Checkpoint must contain a model state dictionary")

    source_model = checkpoint["model"]
    if REMOVED_KEY not in source_model:
        raise KeyError(f"Required classification-head key is missing: {REMOVED_KEY}")
    missing_training_state = [
        key for key in REMOVED_TRAINING_STATE_KEYS if key not in checkpoint
    ]
    if missing_training_state:
        raise KeyError(
            f"Source checkpoint is missing training state: {missing_training_state}"
        )

    # Keep iteration as source provenance. BaseModel.resume_ckpt starts from
    # iteration 0 for a string restore_hint and never consumes this value.
    derived = dict(checkpoint)
    for key in REMOVED_TRAINING_STATE_KEYS:
        del derived[key]
    derived_model = dict(source_model)
    del derived_model[REMOVED_KEY]
    derived["model"] = derived_model

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(derived, temporary)
        verified = load_checkpoint(temporary)
        verified_model = verified["model"]

        removed_checkpoint_keys = sorted(set(checkpoint) - set(verified))
        added_checkpoint_keys = sorted(set(verified) - set(checkpoint))
        if (
            removed_checkpoint_keys != sorted(REMOVED_TRAINING_STATE_KEYS)
            or added_checkpoint_keys
        ):
            raise RuntimeError(
                "Unexpected checkpoint-key difference: "
                f"removed={removed_checkpoint_keys}, added={added_checkpoint_keys}"
            )

        removed = sorted(set(source_model) - set(verified_model))
        added = sorted(set(verified_model) - set(source_model))
        if removed != [REMOVED_KEY] or added:
            raise RuntimeError(
                f"Unexpected model-key difference: removed={removed}, added={added}"
            )
        for key, value in source_model.items():
            if key != REMOVED_KEY and not torch.equal(value, verified_model[key]):
                raise RuntimeError(f"Tensor changed while copying checkpoint: {key}")

        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    source_hash_after = sha256(source)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source checkpoint changed during warm-start preparation")

    print(f"source={source}")
    print(f"source_sha256={source_hash_after}")
    print(f"output={output}")
    print(f"output_sha256={sha256(output)}")
    print(f"source_model_keys={len(source_model)}")
    print(f"derived_model_keys={len(derived_model)}")
    print(f"removed_key={REMOVED_KEY}")
    print(f"removed_training_state={','.join(REMOVED_TRAINING_STATE_KEYS)}")


if __name__ == "__main__":
    main()
