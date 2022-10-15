for pop_size in 1 2 4 8
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/multi_sp/rendezvous.yaml \
        --env_config_file config/envs/rendezvous.yaml \
        --config '{"pop_size": '"${pop_size}"', "save_folder": "results_sweep_rendezvous"}' \
        --env_config '{"mode": "easy"}' --seed $seed
    done
done
