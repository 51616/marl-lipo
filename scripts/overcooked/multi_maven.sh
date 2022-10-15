for seed in 111 222 333 444 555
do
    xvfb-run -a python main.py --config_file config/algs/maven/overcooked.yaml \
    --env_config_file config/envs/overcooked.yaml \
    --config '{"algo_name": "multi_maven", "discrim_coef": 5, "n_iter": 30000, "n_sp_episodes": 4, "pop_size": 8, "render": 0, "save_folder": "training_partners_8", "trainer": "incompat", "z_dim": 8}' --seed $seed
done
