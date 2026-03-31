#!/bin/bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python /data/yhluo/ece567/JaxGCRL/run.py ppo \
  --env ant \
  --backend spring \
  --total_env_steps 10000000 \
  --episode_length 1000 \
  --num_envs 512 \
  --num_eval_envs 256 \
  --num_evals 50 \
  --seed 1 \
  --learning_rate 3e-4 \
  --discounting 0.99 \
  --unroll_length 20 \
  --batch_size 32 \
  --num_minibatches 16 \
  --num_updates_per_batch 4 \
  --normalize_observations \
  --normalize_advantage \
  --clipping_epsilon 0.2 \
  --gae_lambda 0.95 \
  --wandb_project_name exe567-proj1 \
  --wandb_group ant \
  --exp_name ppo_ant
