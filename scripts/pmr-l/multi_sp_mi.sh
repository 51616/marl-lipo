for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/sp_mi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_sp_mi", "discrim_coef": 1, "n_sp_episodes": 400, "n_workers": 16, "pop_size": 1, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/sp_mi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_sp_mi", "discrim_coef": 5, "n_sp_episodes": 400, "n_workers": 16, "pop_size": 2, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/sp_mi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_sp_mi", "discrim_coef": 1, "n_sp_episodes": 400, "n_workers": 16, "pop_size": 4, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' --env_config '{"mode": "hard"}' --seed $seed
done
