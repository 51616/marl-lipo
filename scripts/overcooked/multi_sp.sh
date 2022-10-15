for seed in 111 222 333 444 555
do
    xvfb-run -a python main.py --config_file config/algs/multi_sp/overcooked.yaml \
    --env_config_file config/envs/overcooked.yaml \
    --config '{"pop_size": 8, "render": 0, "save_folder":"training_partners_8"}' --seed $seed
done
