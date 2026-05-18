<div align="center">

# GSFlow-RL

Gaussian-Smoothed Flow Q-Learning for Offline-to-Online Reinforcement Learning

</div>

## Overview

**GSFlow** is an offline-to-online RL algorithm that extends
[Flow Q-Learning (FQL)](https://arxiv.org/abs/2502.02538) with a Gaussian-smoothed
flow policy. It learns a latent VAE over high-value noise samples drawn from
the BC flow model, then distills a Gaussian one-step policy from this prior with
SAC-style entropy regularization. This gives the policy meaningful exploration
during online fine-tuning while preserving the expressivity of the underlying
flow model.

The main implementation is in [`agents/gsflow.py`](agents/gsflow.py).
For comparison, this repository also ships baseline implementations of
**FQL**, **IFQL**, **IQL**, **ReBRAC**, and **SAC** (all under [`agents/`](agents/)),
inherited from the original [FQL repo](https://github.com/seohongpark/fql).

## Installation

Requires Python 3.9+ and JAX. Main dependencies are
`jax >= 0.4.26`, `ogbench == 1.1.0`, and `gymnasium == 0.29.1`.

```bash
pip install -r requirements.txt
```

> **Note:** To use D4RL environments (AntMaze, Adroit), you need to additionally
> set up MuJoCo 2.1.0.

## Quick Start

Train GSflow on an OGBench task with offline-to-online fine-tuning:

```bash
python main.py \
    --env_name=cube-double-play-singletask-v0 \
    --agent=agents/gsflow.py \
    --offline_steps=1000000 \
    --online_steps=1000000 \
    --agent.alpha=10 \
    --offline_alpha=10 \
    --online_alpha=10
```

A minimal end-to-end example is in [`run_example.sh`](run_example.sh).

## Supported Environments

| Family       | Example env name                          | Source     |
|--------------|-------------------------------------------|------------|
| OGBench      | `cube-double-play-singletask-v0`          | `ogbench`  |
| D4RL AntMaze | `antmaze-large-play-v2`                   | `d4rl`     |
| D4RL Adroit  | `pen-human-v1`, `hammer-cloned-v1`, etc.  | `d4rl`     |
| Toy 2D       | `GMM-2D` (multi-modal crescent landscape) | included   |

Environment routing lives in [`envs/env_utils.py`](envs/env_utils.py).

## Running Baselines

Switch agents by pointing `--agent` at a different config file:

```bash
python main.py --env_name=<env> --agent=agents/fql.py     # FQL
python main.py --env_name=<env> --agent=agents/ifql.py    # IFQL
python main.py --env_name=<env> --agent=agents/iql.py     # IQL
python main.py --env_name=<env> --agent=agents/rebrac.py  # ReBRAC
python main.py --env_name=<env> --agent=agents/sac.py     # SAC
```

## Hyperparameter Tips

- **`--agent.alpha`** (BC coefficient): the single most important knob. Tune
  per environment. Defaults to `10.0`.
- **`--offline_alpha` / `--online_alpha`**: GSflow uses these to control
  exploration weight separately in the offline and online phases.
- **`--offline_temperature` / `--online_temperature`**: action sampling
  temperature for the Gaussian policy in each phase.
- **`--agent.latent_dim`** (default `3`): latent dimension of the VAE prior
  over noise. Higher values give richer noise priors at the cost of more
  parameters.
- **`--agent.recon_weight`** (default `1.0`) and **`--agent.kl_weight`**
  (default `0.01`): VAE reconstruction and KL weights.
- **`--agent.num_candidate`** (default `10`): number of candidate noises
  evaluated when selecting the best noise for the VAE target.

For pixel-based OGBench tasks, set `--agent.encoder=impala_small`,
`--p_aug=0.5`, and `--frame_stack=3`.

## Logging

Training logs are written under `--save_dir/<project>/<run_group>/<exp_name>/`
as CSV (`train.csv`, `eval.csv`) and optionally streamed to Weights & Biases
(`--is_wandb=True`).

## Acknowledgments

This codebase is built directly on top of the open-source release of
[Flow Q-Learning](https://github.com/seohongpark/fql) by Park et al. The
baseline agent implementations, training loop, and environment wrappers are
adapted from that repository.

## License

MIT. See [`LICENSE`](LICENSE).
