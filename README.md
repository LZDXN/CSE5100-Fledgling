# CSE5100-Fledgling
Getting things off the ground.

## About this project

**Fledgling** is a reinforcement learning research project focused on dynamic
system identification and control-system design for flying vehicles. The
long-term goal is to develop an end-to-end learning-based process that can
generate autopilot controllers for known or novel aircraft without relying
entirely on traditional manual control-system design.

The pipeline trains separate Proximal Policy Optimization (PPO) agents on a
decoupled per-axis state-space representation of a generic 6 degree-of-freedom
multirotor plant linearized about hover. Each agent produces per-rotor thrust
commands that close the feedback loop on its respective axis. A Linear
Quadratic Integral (LQI) optimal controller derived through randomized search
on the same plant serves as the quantitative baseline.

## Project goals

- Train a per-axis PPO controller that maps observed state feedback to
  continuous per-rotor thrust commands, achieving stable command tracking
  on the decoupled hover-linearized plant.
- Compare against an LQI baseline derived from the same plant.
- Demonstrate that the same training process transfers across rotor counts
  (4, 6, 8) with no manual re-tuning beyond the actor's output dimension.
- Probe robustness via injected disturbances (process noise, actuator noise,
  external sinusoidal force, single-rotor failure) and partial observability
  (state masking, observation noise, observation delay, frame-stacking).

## Repository layout

```
multiRotorPlant.py        6DOF state-space plant + per-axis decomposition
nnController.py           ActorMLP / CriticMLP (Gaussian policy, value head)
ppoTrainer.py             PPO-clip + GAE
nnTrainingLoop.py         Per-axis rollout / update / eval loop
humanLQI.py               Random-search LQI baseline
disturbance.py            Robustness injection (Section 5.3 ablations)
observability.py          Partial-observation regimes (Section 5.4 ablations)
auxEval.py                Cross-condition deterministic policy evaluator
trainingArtifacts.py      Checkpoint / log / summary writers
utilsPlots.py, lqiPlots.py  Live + static plotting
main.py                   Single-axis training entry point
scripts/                  Sweep runners and aggregation
```

# Getting started

## 1. Environment

This project uses Conda for environment management.

```bash
conda env create -f environment.yml
conda activate fledgling-rl
pip install -r requirements-rl.txt
```

If you hit `OMP: Error #15` when importing torch on macOS, export
`KMP_DUPLICATE_LIB_OK=TRUE` (already set inside `main.py`).

## 2. Baseline training

Train per-axis PPO controllers against the linearized hover-trim plant and
produce the LQI baseline for comparison:

```bash
# All four axes (PITCHLON, ROLLLAT, VERT, YAWHDG), four-rotor vehicle
python main.py --exp_name baseline --n_rotors 4 --n_batches 200

# Single axis only (e.g. vertical hover/altitude tracking)
python main.py --axis VERT --exp_name vert_only --n_batches 200

# Skip the LQI baseline derivation (faster iteration during development)
python main.py --axis VERT --no_lqr --n_batches 200
```

Outputs land in `data/<timestamp>_<exp_name>/<Axis>/` and include the actor /
critic checkpoints, training-log CSV, live + final tracking plots, and a
`summary.json` with rise time, settling time, percent overshoot/undershoot,
and final tracking error for the best-eval policy.

Common knobs:

| Flag                 | Default | Meaning                                       |
|----------------------|---------|-----------------------------------------------|
| `--n_rotors`         | 4       | Vehicle rotor count (4, 6, 8 supported)       |
| `--n_batches`        | 200     | Outer PPO batches                             |
| `--batch_size`       | 2048    | Environment steps per batch                   |
| `--max_steps`        | 500     | Max steps per episode (early-term aside)      |
| `--r_cmd`            | 1.0     | Step-command reference value                  |
| `--lr`               | 3e-4    | Adam learning rate                            |
| `--gamma`/`--lam`    | 0.99/0.95 | Discount / GAE bias-variance trade-off      |
| `--seed`             | -1      | Non-negative seeds numpy/torch/random         |

## 3. Auxiliary experiments

Two ablations beyond the baseline are wired into the same training pipeline
and exposed via `main.py` flags. All flags default to no-op, so omitting them
reproduces the baseline run.

### 3a. Robustness to disturbance and noise (Section 5.3)

| Flag                         | Effect                                                |
|------------------------------|-------------------------------------------------------|
| `--enable_process_noise`     | Additive Gaussian state noise per step (Peng et al. 2018) |
| `--enable_actuator_noise`    | Additive Gaussian per-rotor command noise             |
| `--enable_force_disturbance` | Per-episode sinusoidal external force (wind-gust analog) |
| `--rotor_failure_prob 0.2`   | Probability of zeroing one rotor at episode start (Sharma et al. 2021) |

Single training run with all disturbance channels on:

```bash
python main.py --axis VERT --no_lqr --n_batches 100 --exp_name robust_train \
    --enable_process_noise --enable_actuator_noise --enable_force_disturbance \
    --rotor_failure_prob 0.2
```

Full sweep (2×2 train/eval ablation + rotor-failure across n_rotors ∈ {4, 6, 8}):

```bash
python scripts/runRobustnessExperiments.py \
    --out_dir data/aux_robustness \
    --seeds 3 --n_batches 60
```

The sweep writes `robustness_2x2.json`, `robustness_failure.json`,
`robustness_all.json`, and a human-readable `robustness_summary.md` into the
output directory.

### 3b. Restricted observability (Section 5.4)

| Flag                       | Effect                                                  |
|----------------------------|---------------------------------------------------------|
| `--obs_keep_idxs "0,2"`    | Restrict obs to listed state indices (e.g. drop position) |
| `--obs_noise_sigma 0.05`   | Additive Gaussian noise on the obs                      |
| `--obs_delay_steps 3`      | Show the controller a delayed observation               |
| `--obs_history_len 4`      | Frame-stack the last K obs into the actor input (Mnih et al. 2015) |

Single training run with inertial-only obs and frame-stack recovery:

```bash
python main.py --axis VERT --no_lqr --n_batches 100 --exp_name obs_inertial \
    --obs_keep_idxs "0,2" --obs_history_len 4
```

Full sweep across the 7 named regimes (full, inertial-only, position-only,
noisy, delayed, plus stacked-recovery for inertial and delayed):

```bash
python scripts/runObservabilityExperiments.py \
    --out_dir data/aux_observability \
    --seeds 3 --n_batches 60
```

The sweep writes `observability_results.json` and `observability_summary.md`.

### 3c. Aggregating and plotting

Render publication-style bar charts from the aggregated JSON files:

```bash
python scripts/plotAuxiliaryResults.py \
    --robustness_json    data/aux_robustness/robustness_all.json \
    --observability_json data/aux_observability/observability_results.json \
    --out_dir            data/aux_plots
```

This produces:

- `robustness_2x2.png`        — 2×2 train/eval reward + tracking error
- `robustness_failure.png`    — rotor-failure across rotor counts
- `observability_regimes.png` — partial-observability regime comparison

## 4. Reproducing the report

To reproduce the full Section 5.3 / 5.4 result tables at the report's stated
protocol (3 seeds × 60 batches × 2048 steps × 4 rotors on a single axis):

```bash
# ~30 min total on a modern laptop CPU (4 cores)
python scripts/runRobustnessExperiments.py    --seeds 3 --n_batches 60
python scripts/runObservabilityExperiments.py --seeds 3 --n_batches 60
python scripts/plotAuxiliaryResults.py \
    --robustness_json    data/aux_robustness/robustness_all.json \
    --observability_json data/aux_observability/observability_results.json \
    --out_dir            data/aux_plots
```

The runners are parallel-friendly: launch them in separate shells to overlap
the train-bound work, since each `main.py` subprocess holds its own PPO state.
