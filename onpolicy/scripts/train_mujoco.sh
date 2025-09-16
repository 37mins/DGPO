#!/bin/sh
env="mujoco"
scenario="Ant-v2" 
num_agents=1
algo="ours"
exp="exp"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python3 train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} \
    --n_rollout_threads 128 --n_eval_rollout_threads 128 --episode_length 200 \
    --num_mini_batch 16 --num_env_steps 100000000 --use_ReLU \
    --entropy_coef 0.0 --gamma 0.99 0.99 \
    --max_z 2 --div_thresh 1.1 --rex_thresh -2.5 --discri_lr 1e-4 --alpha_lr 0. \
    --wandb_name "cwz19" --user_name "cwz19" --log_interval 1 --eval_interval 5 \
    --use_eval --use_wandb
done