for pop_size in 1 2
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/maven/rendezvous.yaml \
        --env_config_file config/envs/rendezvous.yaml \
        --config '{"discrim_coef": 10, "n_iter": 4000, "n_sp_episodes": 30, "n_workers": 16, "save_folder": "results_sweep_rendezvous", "trainer": "simple", "z_dim": '"${pop_size}"'}' --env_config '{"mode": "easy"}' --seed $seed
    done
done

for pop_size in 4 8
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/maven/rendezvous.yaml \
        --env_config_file config/envs/rendezvous.yaml \
        --config '{"discrim_coef": 1, "n_iter": 4000, "n_sp_episodes": 30, "n_workers": 16, "save_folder": "results_sweep_rendezvous", "trainer": "simple", "z_dim": '"${pop_size}"'}' --env_config '{"mode": "easy"}' --seed $seed
    done
done
