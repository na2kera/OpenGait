# Docker setup

This document describes how to run OpenGait in Docker. The image is based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`, and Python runtime dependencies are installed from `requirements-docker.txt` at build time.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- For GPU training/evaluation: a Linux host with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 1. Build the image

Run this command from the repository root:

```bash
docker compose build
```

## 2. Start a container

```bash
docker compose run --rm opengait bash
```

To keep the service running in the background:

```bash
docker compose up -d
docker exec -it opengait bash
```

The working directory is `/workspace`, where the repository is mounted.

By default, the host `./datasets` directory is mounted to `/data/gait` in the container. To mount a different dataset directory, set `OPENGAIT_DATA_DIR`:

```bash
OPENGAIT_DATA_DIR=/path/to/gait/datasets docker compose run --rm opengait bash
```

For example, if CASIA-B pkl files are under `/data/gait/CASIA-B-pkl` in the container, set `dataset_root: /data/gait/CASIA-B-pkl` in the target config file.

## 3. Check the environment inside the container

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
python -c "import cv2, kornia, einops, yaml; print('deps ok')"
ls /data/gait/CASIA-B-pkl | head
```

## 4. Training and evaluation

See [2.prepare_dataset.md](2.prepare_dataset.md) for dataset preparation.

Single-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 opengait/main.py \
  --cfgs ./configs/gaitbase/gaitbase_da_casiab.yaml --phase train
```

Multi-GPU commands are the same as the Train / Test examples in [0.get_started.md](0.get_started.md).

## Updating dependencies

Rebuild the image after changing `requirements-docker.txt`.

```bash
docker compose build --no-cache
```

## Troubleshooting

| Symptom | Action |
|------|------|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA driver / Container Toolkit is not available on the host. Try a CPU-only setup or run on a GPU Linux host. |
| Zombie processes after DDP exits | Run `sh misc/clean_process.sh` in the container. See [0.get_started.md](0.get_started.md). |
| Shared memory errors | Increase `shm_size` in `docker-compose.yml`. |
