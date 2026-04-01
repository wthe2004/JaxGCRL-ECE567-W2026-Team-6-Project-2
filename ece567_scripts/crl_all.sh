#!/bin/bash
# Reproduce CRL baseline on JaxGCRL
set -e
cd "$(dirname "$0")/.."

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

COMMON="--num_evals 50 --batch_size 256 --discounting 0.99 --action_repeat 1 --unroll_length 62 --min_replay_size 1000 --max_replay_size 10000 --contrastive_loss_fn bwd_infonce --energy_fn norm --train_step_multiplier 1 --no-log_wandb"

run_exp() {
  local env=$1
  local seed=$2
  local num_envs=$3
  local steps=$4
  local ep_len=$5
  echo "=== $env seed=$seed ==="
  python run.py crl \
    --wandb_project_name crl_reproduce --wandb_group "$env" \
    --exp_name "crl_${env}" --seed "$seed" \
    --env "$env" --num_envs "$num_envs" --episode_length "$ep_len" \
    --total_env_steps "$steps" \
    $COMMON \
    2>&1 | tee "ece567_scripts/logs/${env}_s${seed}.log"
  echo "=== Done: $env seed=$seed ==="
}

# Locomotion: ant (512 envs, 10M steps, episode_length=1000)
for seed in 1 2 3; do
  run_exp ant $seed 512 10000000 1000
done

# Manipulation: arm_reach (256 envs, 10M steps, episode_length=1000)
for seed in 1 2 3; do
  run_exp arm_reach $seed 256 10000000 1000
done

# Manipulation: arm_push_easy (256 envs, 10M steps, episode_length=1000)
for seed in 1 2 3; do
  run_exp arm_push_easy $seed 256 10000000 1000
done

echo "=== ALL EXPERIMENTS COMPLETE ==="
