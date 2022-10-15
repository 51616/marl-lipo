for seed in 111 222 333 444 555
do
    xvfb-run -a python main.py --config_file config/algs/sp_mi/overcooked.yaml \
    --env_config_file config/envs/overcooked.yaml \
    --config '{"algo_name": "multi_sp_mi", "discrim_coef": 5, "n_sp_ts": 20000, "pop_size": 8, "render": 0, "save_folder": "training_partners_8", "trainer": "incompat", "z_dim": 8}' \
    --env_config '{}' --seed $seed
done
