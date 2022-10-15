for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/maven/one_step_matrix.yaml \
    --env_config_file config/envs/one_step_matrix.yaml \
    --config '{"algo_name": "multi_maven", "discrim_coef": 5, "n_sp_episodes": 800, "n_workers": 16, "pop_size": 1, "save_folder": "results_sweep_one_step_matrix_k_8", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
    --env_config '{"k": 8}' --seed $seed
done

for pop_size in 2 4 8
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/maven/one_step_matrix.yaml \
        --env_config_file config/envs/one_step_matrix.yaml \
        --config '{"algo_name": "multi_maven", "discrim_coef": 1, "n_sp_episodes": 800, "n_workers": 16, "pop_size": '"${pop_size}"', "save_folder": "results_sweep_one_step_matrix_k_8", "trainer": "incompat", "vary_z_eval": 1, "z_dim": 8}' \
        --env_config '{"k": 8}' --seed $seed
    done
done