for pop_size in 8 16 32 64
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/sp_mi/one_step_matrix.yaml \
        --env_config_file config/envs/one_step_matrix.yaml \
        --config '{"discrim_coef": 50, "n_sp_episodes": 6400, "n_workers": 16, "save_folder": "results_sweep_one_step_matrix_k_8", "trainer": "simple", "z_dim": '"${pop_size}"'}' \
        --env_config '{"k": 8}' --seed $seed
    done
done
