for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/maven/one_step_matrix.yaml \
    --env_config_file config/envs/one_step_matrix.yaml \
    --config '{"discrim_coef": 1, "n_sp_episodes": 6400, "n_workers": 16, "save_folder": "results_sweep_one_step_matrix_k_8", "trainer": "simple", "z_dim": 8}' \
    --env_config '{"k": 8}' --seed $seed
done

for pop_size in 16 32
do
    for seed in 111 222 333
    do
        xvfb-run -a python main.py --config_file config/algs/maven/one_step_matrix.yaml \
        --env_config_file config/envs/one_step_matrix.yaml \
        --config '{"discrim_coef": 5, "n_sp_episodes": 6400, "n_workers": 16, "save_folder": "results_sweep_one_step_matrix_k_8", "trainer": "simple", "z_dim": '"${pop_size}"'}' \
        --env_config '{"k": 8}' --seed $seed
    done
done

for seed in 111 222 333
do
    xvfb-run -a python main.py --config_file config/algs/maven/one_step_matrix.yaml \
    --env_config_file config/envs/one_step_matrix.yaml \
    --config '{"discrim_coef": 10, "n_sp_episodes": 6400, "n_workers": 16, "save_folder": "results_sweep_one_step_matrix_k_8", "trainer": "simple", "z_dim": 64}' \
    --env_config '{"k": 8}' --seed $seed
done
