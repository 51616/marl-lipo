for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/incompat/one_step_matrix.yaml \
    --env_config_file config/envs/one_step_matrix.yaml \
    --config '{"num_xp_pair_sample": 64, "pop_size": 8, "save_folder": "results_sweep_one_step_matrix_k_8", "xp_coef": 0.5}' \
    --env_config '{"k": 8}' --seed $seed
done

for pop_size in 16 32 64
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/incompat/one_step_matrix.yaml \
        --env_config_file config/envs/one_step_matrix.yaml \
        --config '{"num_xp_pair_sample": 64, "pop_size":'"${pop_size}"', "save_folder": "results_sweep_one_step_matrix_k_8", "xp_coef": 1}' \
        --env_config '{"k": 8}' --seed $seed
    done
done
