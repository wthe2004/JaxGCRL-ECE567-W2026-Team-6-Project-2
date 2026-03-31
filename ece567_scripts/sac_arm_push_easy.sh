#!/bin/bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python /data/yhluo/ece567/JaxGCRL/run.py sac \
  --env arm_push_easy \
  --backend mjx \
  --total_env_steps 10000000 \
  --episode_length 1000 \
  --num_envs 256 \
  --num_eval_envs 256 \
  --num_evals 50 \
  --seed 1 \
  --learning_rate 3e-4 \
  --discounting 0.99 \
  --batch_size 256 \
  --unroll_length 62 \
  --min_replay_size 1000 \
  --max_replay_size 10000 \
  --tau 0.005 \
  --h_dim 256 \
  --n_hidden 2 \
  --normalize_observations \
  --use_her \
  --wandb_project_name exe567-proj1 \
  --wandb_group arm_push_easy \
  --exp_name sac_her_arm_push_easy
