for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/trajedi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"diverse_coef": 50, "kernel_gamma": 0.1, "pop_size": 1, "save_folder": "results_sweep_rendezvous"}' \
    --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/trajedi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"diverse_coef": 50, "kernel_gamma": 0, "pop_size": 2, "save_folder": "results_sweep_rendezvous"}' \
    --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/trajedi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"diverse_coef": 5, "kernel_gamma": 0.1, "pop_size": 4, "save_folder": "results_sweep_rendezvous"}' \
    --env_config '{"mode": "hard"}' --seed $seed
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/trajedi/rendezvous.yaml \
    --env_config_file config/envs/rendezvous.yaml \
    --config '{"diverse_coef": 5, "kernel_gamma": 0.1, "pop_size": 8, "save_folder": "results_sweep_rendezvous"}' \
    --env_config '{"mode": "hard"}' --seed $seed
done
