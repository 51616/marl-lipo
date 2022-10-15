for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/trajedi/one_step_matrix.yaml \
    --env_config_file config/envs/one_step_matrix.yaml \
    --config '{"diverse_coef": 0.05, "pop_size": 8, "save_folder": "results_sweep_one_step_matrix_k_8"}' \
    --env_config '{"k": 8}' --seed $seed
done

for pop_size in 16 32 64
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/trajedi/one_step_matrix.yaml \
        --env_config_file config/envs/one_step_matrix.yaml \
        --config '{"diverse_coef": 0.01, "pop_size": '"${pop_size}"', "save_folder": "results_sweep_one_step_matrix_k_8"}' \
        --env_config '{"k": 8}' --seed $seed
    done
done
