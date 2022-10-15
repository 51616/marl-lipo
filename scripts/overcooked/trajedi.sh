for seed in 111 222 333 444 555
do
    xvfb-run -a python main.py --config_file config/algs/trajedi/overcooked.yaml \
    --env_config_file config/envs/overcooked.yaml \
    --config '{"diverse_coef": 5, "kernel_gamma": 0.5, "pop_size": 8, "render": 0, "save_folder": "training_partners_8"}' --env_config '{}' --seed $seed
done
