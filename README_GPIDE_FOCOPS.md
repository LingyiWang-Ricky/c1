# GPIDE + FOCOPS integration notes

This checkout includes a self-contained implementation of the two requested innovation points on top of the AirSim UAV navigation baseline.

## What was added

1. **GPIDE temporal encoder** (`scripts/utils/sequence_gpide.py`):
   - Encodes observation history with PID-inspired operations: observation differencing, summation heads, exponential-smoothing heads, and optional attention heads.
   - Supports both `perception = vector` and `perception = depth`.  Depth images are converted to compact obstacle-sector features plus UAV state, so training is faster than flattening the full image.
   - Trains with a local sequence SAC implementation and saves `model_sequence_gpide.pt` plus checkpoints.

2. **FOCOPS-inspired policy-space constraint** (`scripts/utils/sequence_gpide.py`):
   - Adds obstacle/action/yaw cost critics.
   - Uses a Lagrange multiplier `nu` to penalize policies that exceed the configured per-step cost limit.
   - Adds a KL trust-region penalty against a delayed policy snapshot to avoid unstable policy jumps.

3. **Environment safety signals** (`gym_env/gym_env/envs/airsim_env.py`):
   - `info` now contains `constraint_cost`, `obstacle_cost`, `action_cost`, `yaw_error_cost`, `distance_to_goal`, and `min_distance_to_obstacles`.
   - `reward_final` now accepts optional shaping parameters in `[reward]` while keeping old configs backward-compatible.

4. **Ready-to-run configs**:
   - `configs/config_Trees_SimpleMultirotor_GPIDE_FOCOPS.ini`
   - `configs/config_City_400_Multirotor_2D_GPIDE_FOCOPS.ini`

## Training

From the project root:

```bash
pip install -r requirements.txt
pip install -e gym_env
python scripts/train.py --config config_Trees_SimpleMultirotor_GPIDE_FOCOPS
```

or, with the plotting UI:

```bash
python scripts/start_train_with_plot.py --config configs/config_Trees_SimpleMultirotor_GPIDE_FOCOPS.ini
```

## Evaluation

```bash
python scripts/start_evaluate_with_plot.py --eval_path logs/Trees/<run_folder>
```

The evaluation script detects `temporal_encoder = gpide` and loads `model_sequence_gpide.pt` automatically when present.

## Important parameters

- `[GPIDE] seq_len`: history length.
- `[GPIDE] exp_smoothing_alphas`: exponential-smoothing heads; larger values emphasize recent observations.
- `[FOCOPS] cost_limit`: target per-step safety cost. Lower values are more conservative.
- `[FOCOPS] eta` and `kl_coef`: policy-space trust-region strength.
- `[constraint] safe_distance`: distance at which obstacle cost begins.
- `[safety] enabled`: optional action shield for exploration in dense obstacles.
