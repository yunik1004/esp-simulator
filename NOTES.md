# Reproduction notes: Gridworld λ sweep (Fig. 4), reduced scale

This fork reruns the Gridworld experiment from Park et al., "Adversarial Reinforcement Learning Framework for ESP
Cheater Simulation" ([arXiv 2509.24274v2](https://arxiv.org/abs/2509.24274)), on one desktop machine with a small
fraction of the paper's compute. The target is Fig. 4: detector AP and AUROC, average reward and average trajectory
length as functions of the adversarial coefficient λ.

**Result.** Every Fig. 4 trend has the same direction here, in both seeds. As λ goes from 0.01 to 0.3 to 3, AP and AUROC
fall, reward falls and trajectory length rises. The absolute values do not match the paper. The base agents were
trained for 3–5% of the paper's pretraining budget and are far from converged. The adversarial numbers are
training-time metrics, not a held-out evaluation (see [Caveats](#caveats)).

- Base commit: upstream [`yunik1004/esp-simulator`](https://github.com/yunik1004/esp-simulator) at `68f1ca4`
  (2025-09-05). No training or evaluation code was changed.
- Runs: 2026-10-16 to 2026-10-27. All times below are UTC.
- Raw logs, checkpoints and TensorBoard events are not in the repo (`output/` is gitignored). They are available on
  request.

## Setup

- **Machine:** AMD Ryzen 7 9800X3D (8 cores, 16 threads), AMD Radeon RX 7900 XTX (24 GB), 32 GB RAM shared with
  other services.
- **Container:** everything ran in an Ubuntu 24.04 LXC container with a 16 GB RAM limit. It had 14 threads during
  pretraining. After an unplanned restart on 2026-10-17 00:36 it had 8 threads, for the detector and all adversarial
  runs.
- **Devices:** PPO (pretraining, mixed agent, data generation, adversarial phase) ran on the CPU. Detector training and
  evaluation ran on the GPU (ROCm 6.2.4).
- **Software:** Python 3.12.3, torch 2.6.0+rocm6.2.4, Stable-Baselines3 2.5.0, Lightning 2.5.0.post0, uv 0.13.1.

### Deviations from the paper

| | Paper (§4, App. B) | Here |
|---|---|---|
| Environments per run | 64 | 4 |
| Samples per PPO update | 64 × 2048 = 131,072 | 4 × 2048 = 8,192 |
| Pretraining | 1e9 steps, checkpoint with the best reward | stopped at 74.5–80.0M; used 32.8M (seed 0) and 44.8M (seed 1), see [Pretraining](#pretraining) |
| Honest checkpoint | best reward | same step as that seed's cheater checkpoint |
| Adversarial phase | 4e8 steps, final checkpoint | compared at 60M and 155M (runs stopped at 158–329M) |
| λ | 0.01, 0.03, 0.1, 0.3, 1, 3, 10 | 0.01, 0.3, 3 |
| Settings | Joint and Cheater-only | Joint only |
| Seeds | 3 | 2 (0 and 1) |
| Adversarial metrics | final checkpoint (split and game count not stated) | rolling mean over training rollouts, see [How the adversarial numbers are measured](#how-the-adversarial-numbers-are-measured) |

Gradient steps per sample are the same as in the paper (4 epochs, minibatch 1024), but each update draws on 4
environments instead of 64.

Everything else is the repository default, which matches App. B:
- environment: 11 × 11 grid, 3 × 3 view, 5 items;
- PPO: `n_steps` 2048, batch 1024, 4 epochs, lr 3e-4, entropy 0.01, γ = 1;
- trajectory detector: batch 8, lr 3e-4, 100 epochs, checkpoint with the lowest validation loss;
- detector data: 10k / 2k / 2k games per policy;
- reward-shaping penalty.

### Changes to the repository

- `pyproject.toml` / `uv.lock`: `torch==2.6.0` and `pytorch-triton-rocm==3.2.0` come from the PyTorch ROCm 6.2.4 index
  on Linux. The lock has no `nvidia-*` packages. Other platforms still resolve plain `torch==2.6.0`.
- `.python-version`: 3.12, because the torch 2.6.0 ROCm wheels stop at CPython 3.13 and uv would otherwise pick 3.14.
- `run_with_heartbeat.py`: an optional wrapper that sends a heartbeat to an Uptime Kuma push monitor while a command
  runs. It was used for pretraining only and does not affect training.
- `NOTES.md`: this file.

Notes on running the upstream code:
- **Install:** the README's `uv pip install .` installs only the empty top-level `esp_simulator/__init__.py`, because
  `[tool.setuptools] packages` lists only `esp_simulator`. The scripts then fail with
  `ModuleNotFoundError: esp_simulator.policies`. `uv pip install -e .` works. `uv sync` removes that editable install,
  so run `.venv/bin/python` directly rather than `uv run`.
- **Detector data names:** the README writes `train_data.npy`, but `train_classifier.py` takes separate
  `*_noncheater.npy` / `*_cheater.npy` files per split. The files here follow the classifier's flags.
- **`--num_processes` defaults to 64** in the shared parser (`scripts/grid_world/utils/agents.py`). This crashed the
  container with `construct_mixed_agent.py`, which builds 3 vectorised environments × 64 processes. It works with
  `--num_processes 1`.
- **iGPU:** the 9800X3D's integrated GPU is visible to ROCm. Lightning's default `devices="auto"` then starts DDP across
  both GPUs and fails on the iGPU. Setting `ROCR_VISIBLE_DEVICES` to the discrete GPU's UUID fixes it. The
  `hipBLASLt ... unsupported architecture` warning on gfx1100 is harmless.

### Timeline (UTC)

| When | What |
|---|---|
| 10-16 11:39 → 23:01 | Pretraining, 4 runs at once (11.4 h) |
| 10-16 23:01 → 23:14 | Checkpoint selection, convergence check |
| 10-17 00:46 | Mixed agents built (after an out-of-memory restart, see above) |
| 10-17 01:03 → 01:37 | Detector data |
| 10-18 03:28 → 04:03 | Detector training and test |
| 10-18 04:43 → 10-23 22:27 | Joint, seed 0, λ 0.01 / 0.3 / 3 at once |
| 10-23 22:32 → 10-27 03:59 | Joint, seed 1, λ 0.01 / 0.3 / 3 at once |

## Pretraining

Honest and pure-cheater agents for seeds 0 and 1 were trained at the same time, with 4 environments each, until they
were stopped at 74.5–80.0M steps.

### The pure cheater collapsed in both seeds

From `rollout/ep_len_mean` and `rollout/ep_rew_mean` (rolling means over 100 episodes, read every 1M steps):

| | Best phase | Before the collapse | Collapse |
|---|---|---|---|
| Cheater, seed 0 | 25–36M: 92–140 steps, reward 4.09–4.44 | 37–49M: 143–245 steps, reward 4.08–4.38 | 385 steps at 50M; 484 from 53M |
| Cheater, seed 1 | 37–49M: 62–109 steps, reward 4.43–4.61 | (none) | 409 steps at 50M; 484 from 52M |

484 steps is the episode limit (4 × 11²), so a policy at 484 never finishes by collecting all the items. Reward stayed
at 2.9–3.8 until the runs were stopped. The honest runs did not collapse. At the end (78.8M and 78.6M) they were at
89 and 103 steps, with reward 4.54 and 4.52.

Both seeds started to fail around 50M steps. I don't know why. One candidate is the 16× smaller rollout per update;
I have not tested it.

### Checkpoint selection

Because of the collapse, the latest checkpoints could not be used. Checkpoints are saved every 200k steps. Candidates
around each seed's best training window were played on validation seeds 10000–10499 (500 games) with
`evaluate_agent.py`:

| Checkpoint | Steps | Reward (mean ± std) |
|---|---|---|
| Cheater s0, 26.8M | 137.3 | 4.204 ± 0.628 |
| Cheater s0, 28.8M | 122.3 | 4.264 ± 0.600 |
| Cheater s0, 30.8M | 110.5 | 4.289 ± 0.604 |
| **Cheater s0, 32.8M** | **105.2** | **4.375 ± 0.514** |
| Cheater s0, 34.8M | 107.2 | 4.378 ± 0.459 |
| Cheater s1, 40.8M | 89.2 | 4.503 ± 0.442 |
| Cheater s1, 42.8M | 96.2 | 4.488 ± 0.492 |
| **Cheater s1, 44.8M** | **80.3** | **4.524 ± 0.436** |
| Cheater s1, 46.8M | 82.1 | 4.537 ± 0.421 |
| Honest s0, 32.8M | 153.5 | 4.174 ± 0.536 |
| Honest s1, 44.8M | 144.7 | 4.316 ± 0.443 |

In each seed the two best rewards are within one standard error of each other (about 0.03). The tie was broken toward
the earlier checkpoint, which is further from the collapse. The chosen checkpoints also have the fewest steps. The
honest checkpoint is taken at the same step, so both agents of a seed have had the same amount of training.

### Convergence check on test seeds

Test seeds 12000–13999 (2,000 games), selected checkpoints. Paper values are from Table 1.

| | Cheater reward | Honest reward | Difference | Difference / SE | Cheater steps | Honest steps |
|---|---|---|---|---|---|---|
| Seed 0 (32.8M) | 4.365 ± 0.526 | 4.207 ± 0.531 | +0.159 | 9.5 | 105.8 | 148.0 |
| Seed 1 (44.8M) | 4.539 ± 0.412 | 4.298 ± 0.498 | +0.241 | 16.7 | 78.0 | 150.5 |
| Paper | 4.759 ± 0.011 | 4.676 ± 0.005 | +0.083 | | 45.7 | 62.1 |

At matched steps the cheater has a higher reward and shorter episodes than the honest agent, as in the paper. Both
agents are far below the paper's reward and take 1.7–2.4 times as many steps.

The advantage depends on the matched steps. On the same test seeds, checkpoints from the end of the runs gave:
- cheaters: 483.8 and 484.0 steps, reward 3.50 and 3.70 (collapsed);
- honest agents: 99.6 and 93.7 steps, reward 4.490 and 4.534.

The late honest agent beats the selected seed-0 cheater (4.490 vs 4.365) and ties the seed-1 cheater (4.534 vs 4.539).
The honest agent kept improving after the cheater collapsed.

## Detector

Data came from `gendata_classifier.py` with each seed's selected checkpoints. Per policy and seed there are 10,000
training games (seeds 0–9999), 2,000 validation games (10000–11999) and 2,000 test games (12000–13999). Each sample is
2 × 11 × 11 (movement heatmap and initial board). No sample appears in more than one split. The mean episode lengths of
the test data equal the test-seed numbers above.

The trajectory detector was trained for the default 100 epochs. Validation loss reached its minimum early and then
rose steadily:
- seed 0: 0.511 after epoch 0, minimum 0.384 at epoch 7, 1.680 at epoch 99;
- seed 1: 0.466 after epoch 0, minimum 0.394 at epoch 8, 2.186 at epoch 99.

As in App. B, the checkpoint with the lowest validation loss was kept.

Against the pure cheater, test split:

| | AP | AUROC | Test loss |
|---|---|---|---|
| Seed 0, epoch 7 (selected) | 0.889 | 0.898 | 0.409 |
| Seed 0, epoch 99 | 0.863 | 0.875 | 1.815 |
| Seed 1, epoch 8 (selected) | 0.877 | 0.899 | 0.394 |
| Seed 1, epoch 99 | 0.859 | 0.881 | 2.186 |
| Paper (Table 1) | 0.772 ± 0.012 | 0.810 ± 0.011 | |

Here the detector separates the pure cheater from the honest agent more easily than in the paper. The two agents here
also differ more: 78–106 vs 148–150 steps, against 45.7 vs 62.1 in the paper. That is a plausible reason, but I have
not tested it.

## Adversarial training (Joint)

Setup:
- **Agents and detector:** each seed uses its own mixed agent (`construct_mixed_agent.py`, built from the selected
  cheater and honest checkpoints), its own honest checkpoint and its own selected detector.
- **λ and settings:** λ ∈ {0.01, 0.3, 3} with `--train_classifier` (Joint) and the default `--method reward_shaping`.
- **Penalty:** at the end of each episode the reward is reduced by λ · softplus(logit) = −λ · log(1 − D(τ)), as in
  Eq. 5.
- **Detector updates:** the detector is updated once per PPO iteration (1 epoch, batch 8, lr 3e-4).
- **Batches:** each run used about 3.5 GB of RAM, so the three λ runs of one seed ran together.
- **Steps reached:** seed 0 reached 279.9M / 329.1M / 328.7M (λ 0.01 / 0.3 / 3) and seed 1 reached 157.6M / 166.5M /
  176.5M. Results are compared at 60M and at 155M, the furthest step all six runs reached.

### How the adversarial numbers are measured

These are the `adversary/` and `rollout/` values that `train_adversarial_agent.py` logs during training. They are not
a separate evaluation.
- **AP and AUROC:** in each PPO iteration (8,192 steps) the detector, as it is at that moment, scores every episode the
  agent finishes. It also scores an equal number of fresh honest-agent episodes. AP and AUROC are computed on that
  pooled set. The logged value is the mean over the last 100 iterations, about 0.8M steps.
- **Episode length and reward:** the usual rolling means over the last 100 episodes. Reward is the game reward
  recorded by SB3's `Monitor` wrapper and does not include the λ penalty.
- **Seeds:** all of these are training episodes, not the held-out test seeds.
- **Precision:** values are shown as logged, to three significant figures.

The paper evaluates final checkpoints. These numbers are not directly comparable to Fig. 4, and a test-seed evaluation
of the checkpoints has not been done yet.

### Results at 60M steps

Seed 0 / seed 1.

| λ | AP | AUROC | Episode length | Reward |
|---|---|---|---|---|
| 0.01 | 0.956 / 0.935 | 0.957 / 0.939 | 109 / 74.3 | 4.38 / 4.55 |
| 0.3 | 0.816 / 0.832 | 0.822 / 0.844 | 112 / 90.5 | 4.30 / 4.52 |
| 3 | 0.719 / 0.652 | 0.718 / 0.659 | 144 / 119 | 4.24 / 4.43 |
| Pure cheater (test seeds) | 0.889 / 0.877 | 0.898 / 0.899 | 105.8 / 78.0 | 4.37 / 4.54 |
| Honest (test seeds) | | | 148.0 / 150.5 | 4.21 / 4.30 |

### Results at 155M steps

| λ | AP | AUROC | Episode length | Reward |
|---|---|---|---|---|
| 0.01 | 0.978 / 0.952 | 0.977 / 0.954 | 64.1 / 76 | 4.56 / 4.54 |
| 0.3 | 0.868 / 0.837 | 0.866 / 0.843 | 101 / 96.9 | 4.35 / 4.48 |
| 3 | 0.678 / 0.730 | 0.667 / 0.741 | 125 / 122 | 4.23 / 4.41 |

The reference rows use the selected pretrained agents and detectors on test seeds, so they were measured differently
from the λ rows.

### What matches Fig. 4

- AP and AUROC fall as λ rises, in both seeds at both points. Mean AP over the two seeds:
  - 60M: 0.946 (λ 0.01), 0.824 (λ 0.3), 0.686 (λ 3);
  - 155M: 0.965, 0.853, 0.704.
- Reward falls and episode length rises with λ, again in both seeds at both points.
- At λ = 0.01 the co-trained detector scores higher (AP 0.94–0.98) than the pretrained detector did against the pure
  cheater (0.88–0.89). The paper describes the same pattern for λ ≤ 0.1. The two measurements differ, as noted above.
- At λ = 3 AP is 0.65–0.73, compared with "around 0.5–0.6" in the paper for λ ≥ 1. This is with far less training.

### What depends on the seed

From 60M to 155M:

| λ | AP change, seed 0 | AP change, seed 1 |
|---|---|---|
| 0.01 | +0.022 | +0.017 |
| 0.3 | +0.052 | +0.005 |
| 3 | −0.041 | +0.078 |

With two seeds I don't draw conclusions about longer training.

One observation: at λ = 0.01, seed 0's episode length fell from 109 at 60M to 64 at 155M, with reward 4.56. That is
shorter and better than either pretrained agent of that seed. I have not looked into why.

## Caveats

- Adversarial metrics are rolling training-time means with a co-trained detector, not a test-seed evaluation of final
  checkpoints.
- There are two seeds and no error bars. The paper uses three seeds.
- The base agents are under-trained: test-seed reward 4.21–4.54 vs 4.68–4.76 in the paper.
- The pure-cheater checkpoints were chosen before a training collapse, not as the best of a full run.
- Only Joint was run, with three of the seven λ values.
- The thread count dropped from 14 to 8 between pretraining and the adversarial phase. This affects throughput only.

## Throughput and resources

- **Pretraining smoke test:** 4 runs × 4 environments at once, 14 threads, rates over the last ~120 s. On the CPU this
  gave 1,760–1,784 steps/s per run with about 1.5 GB RAM per run. On the GPU it gave 1,265–1,276 steps/s with about
  2.4 GB per run. Pretraining therefore ran on the CPU.
- **Full pretraining:** 74.5–80.0M steps per run in 11.4 h, about 1,820–1,950 steps/s per run.
- **Adversarial phase, 8 threads:**
  - one run alone: 854 steps/s;
  - three runs at once: 566–677 steps/s per run (SB3's cumulative `time/fps` at 60M);
  - λ = 0.01 was the slowest, 10–13% below λ = 3 in the same seed;
  - memory: about 3.5 GB per run.
- **Disk:** checkpoints every 200k steps take about 12 MB each in the adversarial phase. `output/` reached 77 GB.

## Not done yet

- A test-seed evaluation of the adversarial checkpoints at 60M and 155M.
- The Cheater-only setting.
- λ = 0.03, 0.1, 1 and 10.
- A third seed.
- The paper's full budgets.

## Questions for the authors

1. Did your pure-cheater pretraining ever collapse, or need settings beyond App. B? Both seeds here collapsed at
   about 50M steps with 8,192 samples per update.
2. Are the pretrained checkpoints (honest, pure cheater, detector) available?
3. Are App. B's "about 2.5 days" (pretraining) and "about 3 days" (adversarial) per run, or for all runs together?
4. What licence applies to the code? The repository has no LICENSE file.
5. For Fig. 4, which games are the metrics computed on (split and number), and are the error bars the standard
   deviation over the three seeds?

## Reproducing

All commands run from the repository root with `.venv/bin/python`, `OMP_NUM_THREADS=2`, and
`ROCR_VISIBLE_DEVICES=<discrete GPU UUID>` for GPU jobs. S is the seed (0 or 1), N is the selected step (32800000 for
seed 0, 44800000 for seed 1). Runs were stopped with SIGINT to the process group. Some commands below were run
interactively; they use the same scripts and flags as the logs show.

Pretraining (4 runs at once):

```bash
python scripts/grid_world/train_agent.py [--cheater] --seed S --num_processes 4 --accelerator cpu \
  --checkpoint_freq 50000 --log_dir output --log_name {honest,cheater}_sS
```

Selection (seeds 10000–10499) and convergence check (seeds 12000–13999):

```bash
python scripts/grid_world/evaluate_agent.py [--cheater] --accelerator cpu \
  --checkpoint output/{honest,cheater}_sS_1/rl_model_<step>_steps.zip --seed_envs 10000-10499
```

Mixed agent:

```bash
python scripts/grid_world/construct_mixed_agent.py --seed S --accelerator cpu --num_processes 1 \
  --checkpoint output/cheater_sS_1/rl_model_N_steps.zip \
  --checkpoint_normal output/honest_sS_1/rl_model_N_steps.zip --log_dir output --log_name mixed_sS
```

Detector data (12 files per seed; splits train 0-9999, val 10000-11999, test 12000-13999):

```bash
python scripts/grid_world/gendata_classifier.py [--cheater] --accelerator cpu \
  --checkpoint output/{honest,cheater}_sS_1/rl_model_N_steps.zip --seed_envs 0-9999 \
  --output_path output/data_sS/train_{noncheater,cheater}.npy
```

Detector training and test:

```bash
python scripts/grid_world/train_classifier.py --accelerator gpu --seed S --cls_type trajectory --num_workers 4 \
  --log_dir output/cls_sS \
  --dataset_normal_train output/data_sS/train_noncheater.npy --dataset_normal_val output/data_sS/val_noncheater.npy \
  --dataset_esp_train output/data_sS/train_cheater.npy --dataset_esp_val output/data_sS/val_cheater.npy
python scripts/grid_world/evaluate_classifier.py --accelerator gpu --cls_type trajectory --checkpoint <ckpt> \
  --dataset_normal output/data_sS/test_noncheater.npy --dataset_esp output/data_sS/test_cheater.npy
```

Selected detectors: `output/cls_s0/lightning_logs/version_1/checkpoints/epoch=7-step=20000.ckpt` and
`output/cls_s1/lightning_logs/version_0/checkpoints/epoch=8-step=22500.ckpt`.

Joint adversarial training (3 runs at once per seed):

```bash
python scripts/grid_world/train_adversarial_agent.py --mixed --cls_type trajectory --train_classifier \
  --accelerator cpu --seed S --num_processes 4 --n_steps 2048 --adv_coef λ \
  --checkpoint output/mixed_sS_1/rl_model_mixed.zip \
  --checkpoint_normal output/honest_sS_1/rl_model_N_steps.zip \
  --checkpoint_classifier <selected detector> --checkpoint_freq 50000 --log_dir output --log_name joint_sS_lλ
```
