#!/usr/bin/env bash
# Minimal example: train GSflow on an OGBench task with offline-to-online RL.

set -euo pipefail

python main.py \
    --env_name=cube-double-play-singletask-v0 \
    --agent=agents/gsflow.py \
    --offline_steps=1000000 \
    --online_steps=1000000 \
    --agent.alpha=10 \
    --offline_alpha=10 \
    --online_alpha=10 \
    --offline_temperature=0 \
    --online_temperature=1 \
    --seed=42 \
    --save_dir=./exp/ \
    --run_group=gsflow_example \
    --is_wandb=False
