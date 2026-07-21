# SUSTech1K warm-start fine-tuning

This directory contains the preparation utilities for the preregistered
SUSTech1K-to-CASIA-B DeepGaitV2 fine-tuning experiment.

## Prepare the warm-start checkpoint

Run this inside an OpenGait environment with PyTorch installed:

```bash
python dgv2_finetune/prepare_warm_start.py
```

The command verifies the fixed source checkpoint hash, creates a new checkpoint
without `BNNecks.fc_bin` or optimizer/scheduler state, reloads it, and confirms
that this is the only removed model key and these are the only removed top-level
keys. It never overwrites an existing output or modifies the source.

## Generate and validate configs

```bash
python dgv2_finetune/generate_configs.py
python dgv2_finetune/generate_configs.py --check
```

The generated configs are under `configs/deepgaitv2/finetune_sustech/`. Seed 0
is generated for all 13 subsets. Seed 1 is generated only for the two scout
subsets, `000-180` and `nm1-cl1`.

Generation and `--check` both validate each source config (class_num 75,
sampler shape, disabled augmentation) and that its `dataset_partition` has the
same TEST_SET as the canonical `datasets/CASIA-B/CASIA-B.json` (the 49-subject
holdout 076-124). This rejects the OpenGait standard-protocol partitions
(`CASIA-B-20-2.json`, `-40`, `-60`, `-80`), whose 50-subject test set includes
subject 075 and would silently produce incomparable accuracies.

Do not pass `--iter` when starting fine-tuning. It would replace the trainer's
warm-start checkpoint path with a same-run iteration number. For checkpoint
sensitivity evaluation, use `--phase test --iter {5000|10000|20000|30000}`.
A new warm-start launch is rejected when the same `save_name` already has a
matching checkpoint. Resume explicitly, or preserve the existing run and use a
new `save_name`; an ordinary new-run command can never overwrite it silently.
An intentional interrupted-training resume requires both `--iter N` and
`--allow-train-iter-override`; the extra flag prevents accidental replacement
of the warm-start path on a new run. The flag is rejected when used alone, in
test phase, or with a scratch-style integer restore hint. Negative `--iter`
values are also rejected explicitly. This resume path requires and restores
the model strictly at the requested checkpoint iteration, together with the
saved optimizer, scheduler, and AMP GradScaler state, instead of applying the
warm-start reset. The requested iteration must be below `total_iter`; a missing
state or an iteration mismatch is rejected before training continues.

OpenGait checkpoints do not save RNG, sampler, or data-loader state. A resumed
run therefore is not bit-identical to an uninterrupted run even when model,
optimizer, scheduler, AMP GradScaler, and iteration are restored. Record every
recovery; for a result that affects the preregistered conclusion, rerun that
seed from the beginning when practical.

Use the seed-specific generated configs for preregistered runs. A CLI
`--base-seed` override is rejected when it does not match the trailing `-sN` in
`save_name`, preventing one seed from writing into another seed's output path.
