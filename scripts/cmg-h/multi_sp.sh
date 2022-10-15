for pop_size in 8 16 32 64
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/multi_sp/one_step_matrix.yaml \
        --env_config_file config/envs/one_step_matrix.yaml \
        --config '{"pop_size": '"${pop_size}"', "save_folder": "results_sweep_one_step_matrix_uneven_m32"}' \
        --env_config '{}' --seed $seed
    done
done
