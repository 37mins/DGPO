#!/bin/sh
env="mujoco"
scenario="Ant-v2"
num_agents=1
algo="ours"
exp="check"
seed=1


python3 render/render_mujoco.py --save_gifs --env_name ${env} \
--algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
--seed ${seed} --num_agents ${num_agents} --use_ReLU --use_wandb \
--n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 100000 \
--model_dir "results/${env}/${scenario}/${algo}/exp/run2/models" \
--max_z 2 --hidden_size 64