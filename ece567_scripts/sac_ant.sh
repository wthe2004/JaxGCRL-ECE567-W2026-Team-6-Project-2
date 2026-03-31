#!/bin/bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
uv run python /data/yhluo/ece567/JaxGCRL/run.py sac \
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
  --wandb_group ant \
  --exp_name sac_her_ant
