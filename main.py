def wait():
    from coop_marl.utils import input_with_timeout
    try:
        t = 30
        input_with_timeout(f'Press Enter (or wait {t} seconds) to continue...', timeout=t)
    except Exception:
        print('Input timed out, executing the next command.')

def main():

    from coop_marl.utils import pblock, parse_args, create_parser
    args, conf, env_conf, trainer = parse_args(create_parser())
    import sys
    from coop_marl.utils import get_logger, set_random_seed
    logger = get_logger()
    logger.info(pblock(' '.join(sys.argv), 'Argv...'))
    logger.info(pblock(args, 'CLI arguments...'))
    logger.info(pblock(conf, 'Training config...'))
    logger.info(pblock(env_conf, 'Environment config...'))
    # wait()
    set_random_seed(args.seed)
    # wandb.init(project=env_name, name=run_name, dir=conf['save_dir'], mode='offline', resume=True)
    # import wandb
    # wandb.init(project=...,
    #            name=
    #            pytorch=True)

    from tqdm import tqdm
    from coop_marl.trainers import registered_trainers
    trainer = registered_trainers[trainer](conf, env_conf)
    start_iter = trainer.iter
    save_interval = conf.save_interval if conf.save_interval else conf.eval_interval
    for i in tqdm(range(start_iter,conf.n_iter)):
        _ = trainer.train() # collect data and update the agents
        if ((i+1) % conf.eval_interval==0) or ((i+1)==conf.n_iter) or (i==0):
            _ = trainer.evaluate()
        if ((i+1) % save_interval==0) or ((i+1)==conf.n_iter) or (i==0):
            trainer.save()
    try:
        import ray
        ray.shutdown()
        logger.info(f'Ray is shutdown...')
    except Exception as e:
        logger.error(e)
    # wandb.finish()
    logger.close()
    if conf.render:
        import subprocess
        subprocess.run([f'python gif_view.py --path {conf["save_dir"]}'], shell=True)

if __name__=='__main__':
    main()
