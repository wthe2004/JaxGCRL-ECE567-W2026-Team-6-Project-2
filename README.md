# Team 6 Project 2 Phase 1

The original README of this repo is in [original_README.md](original_README.md).

## Scripts

All training scripts are in `ece567_scripts/`. Change `--seed` to run different seeds.

| Script | Baseline | Envs |
|---|---|---|
| `crl_all.sh` | Contrastive RL | ant, arm_reach, arm_push_easy (3 seeds each) |
| `sac_ant.sh` / `sac_arm_reach.sh` / `sac_arm_push_easy.sh` | SAC | per env |
| `ppo_ant.sh` / `ppo_arm_reach.sh` / `ppo_arm_push_easy.sh` | PPO | per env |

`extract_results.py` parses training logs and prints per-seed and aggregated metrics.

## Code Patches Required

If running on newer GPUs (e.g. Blackwell, compute cap 12.0) with JAX >= 0.5, apply these fixes:

1. **`jaxgcrl/agents/crl/crl.py`**: Convert `train_env.goal_indices` to a Python tuple before use inside JIT-compiled functions (JAX >= 0.5 cannot hash traced arrays as static args). Also ensure `params` is assigned before return at the end of `train_fn`.

2. **`jaxgcrl/envs/manipulation/arm_envs.py`**: Resolve relative XML asset paths (e.g. `envs/assets/panda_reach.xml`) to absolute paths using `os.path`, so the code works regardless of working directory.

3. **`jaxgcrl/utils/evaluator.py`**: Skip missing metric keys in the evaluation loop (manipulation envs don't provide all metrics that locomotion envs do).
