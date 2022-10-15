for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/incompat/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"pop_size": 1, "save_folder": "results_sweep_rendezvous", "xp_coef": 0.5}' --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/incompat/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"pop_size": 2, "save_folder": "results_sweep_rendezvous", "xp_coef": 0.25}' --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/incompat/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"pop_size": 4, "save_folder": "results_sweep_rendezvous", "xp_coef": 0.25}' --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/incompat/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"pop_size": 8, "save_folder": "results_sweep_rendezvous", "xp_coef": 0.25}' --env_config '{"mode": "hard"}' --seed $seed
done
