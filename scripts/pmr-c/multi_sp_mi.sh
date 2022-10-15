for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/sp_mi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_sp_mi", "discrim_coef": 1, "n_sp_episodes": 400, "n_workers": 16, "pop_size": 1, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
    --env_config '{"mode": "easy"}' --seed $seed
done

for pop_size in 2 4
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/sp_mi/rendezvous.yaml \
        --env_config_file config/envs/rendezvous.yaml \
        --config '{"algo_name": "multi_sp_mi", "discrim_coef": 1, "n_sp_episodes": 400, "n_workers": 16, "pop_size": '"${pop_size}"', "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
        --env_config '{"mode": "easy"}' --seed $seed
    done
done
