# Docker 環境構築

OpenGait を Docker で使う手順です。ベースイメージは `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` で、学習・評価用の Python 依存はイメージビルド時に `requirements-docker.txt` から入ります。

## 前提

- [Docker](https://docs.docker.com/get-docker/) と [Docker Compose](https://docs.docker.com/compose/install/) が入っていること
- GPU 学習をする場合: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) が入っている Linux ホスト（Mac の Docker Desktop では GPU パススルー不可）

## 1. イメージのビルド

リポジトリルートで:

```bash
docker compose build
```

## 2. コンテナに入る

```bash
docker compose run --rm opengait bash
```

`docker compose up -d` で常駐させる場合:

```bash
docker compose up -d
docker exec -it opengait bash
```

作業ディレクトリは `/workspace`（ホストのリポジトリがマウントされます）。

データセットはホストの `/home/kera/data/gait` をコンテナの `/data/gait` にマウントします（`docker-compose.yml` 参照）。CASIA-B の pkl は `/data/gait/CASIA-B-pkl` を `dataset_root` に指定します。

## 3. 動作確認（コンテナ内）

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
python -c "import cv2, kornia, einops, yaml; print('deps ok')"
ls /data/gait/CASIA-B-pkl | head
```

## 4. 学習・評価

データセットの準備は [2.prepare_dataset.md](2.prepare_dataset.md) を参照。

単 GPU の例:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 opengait/main.py \
  --cfgs ./configs/gaitbase/gaitbase_da_casiab.yaml --phase train
```

複数 GPU の例は [0.get_started.md](0.get_started.md) の Train / Test セクションと同じです。

## 依存の更新

`requirements-docker.txt` を変更したあとは再ビルドしてください。

```bash
docker compose build --no-cache
```

## トラブルシュート

| 症状 | 対処 |
|------|------|
| `could not select device driver "" with capabilities: [[gpu]]` | ホストに NVIDIA ドライバ / Container Toolkit が無い。CPU のみで試すか、GPU 付き Linux サーバで実行 |
| DDP 終了後にゾンビプロセス | コンテナ内で `sh misc/clean_process.sh`（[0.get_started.md](0.get_started.md) 参照） |
| `shm_size` 不足 | `docker-compose.yml` の `shm_size` を増やす |
