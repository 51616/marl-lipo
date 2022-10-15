for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/maven/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_maven", "discrim_coef": 10, "n_iter": 4000, "n_sp_episodes": 30, "n_workers": 16, "pop_size": 1, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
    --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/maven/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_maven", "discrim_coef": 10, "n_iter": 4000, "n_sp_episodes": 30, "n_workers": 16, "pop_size": 2, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
    --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/maven/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"algo_name": "multi_maven", "discrim_coef": 5, "n_iter": 4000, "n_sp_episodes": 30, "n_workers": 16, "pop_size": 4, "save_folder": "results_sweep_rendezvous", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
    --env_config '{"mode": "hard"}' --seed $seed
done
