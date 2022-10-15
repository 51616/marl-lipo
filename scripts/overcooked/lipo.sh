for seed in 111 222 333 444 555
do
    xvfb-run -a python main.py --config_file config/algs/incompat/overcooked.yaml \
    --env_config_file config/envs/overcooked.yaml \
    --config '{"discrim_coef": 0.5, "pop_size": 8, "render": 0, "save_folder": "training_partners_8", "xp_coef": 0.3, "z_dim": 8}' \
    --seed $seed
done