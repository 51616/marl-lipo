import os
import yaml
import argparse
import random
from datetime import datetime

import torch
from yamlinclude import YamlIncludeConstructor

from coop_marl.utils import Dotdict, update_existing_keys, get_logger, pblock

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

DEF_CONFIG = 'def_config'

def save_yaml(conf, path):
    os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    with open(f'{path}.yaml', 'w') as f:
        yaml.dump(conf, f, default_flow_style=False)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--env_config_file', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--config', default={}, type=yaml.load)
    parser.add_argument('--env_config', default={}, type=yaml.load)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--run_name', default='', type=str)
    return parser

def get_def_conf(data, init_call=False):
    if DEF_CONFIG not in data:
        if init_call:
            return {}
        return data
    cur_level = {k:v for k,v in data.items() if k != DEF_CONFIG}
    next_level = get_def_conf(data[DEF_CONFIG])
    next_level.update(cur_level)
    return next_level

def parse_nested_yaml(yaml):
    def_conf = get_def_conf(yaml, True)
    conf = {k:v for k,v in yaml.items() if k != DEF_CONFIG}
    if def_conf is not None:
        def_conf.update(conf)
        conf = def_conf
    return conf
    
def parse_args(parser):
    args = parser.parse_args()
    data = yaml.load(args.config_file, Loader=yaml.FullLoader)
    conf = parse_nested_yaml(data)

    env_conf = yaml.load(args.env_config_file, Loader=yaml.FullLoader)
            
    # replace the config params with hparams from console args
    unused_param = [None, None]
    conf_names = ['config', 'env config']
    for i, (cli_conf, yaml_conf, text) in enumerate(zip([args.config, args.env_config], [conf, env_conf], conf_names)):
        yaml_conf, unused_param[i] = update_existing_keys(yaml_conf, cli_conf)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if len(args.run_name)>0:
        run_name = args.run_name
    conf['run_name'] = run_name

    if not getattr(conf, 'save_dir', ''):
        env_folder = f'{env_conf["name"]}_{env_conf["mode"]}' if 'mode' in env_conf else f'{env_conf["name"]}'
        save_folder = conf['save_folder']
        conf['save_dir'] = f'{save_folder}/{env_folder}/{conf["algo_name"]}/{run_name}'
    logger = get_logger(log_dir=conf['save_dir'], debug=conf['debug'])
    [logger.info(pblock(unused_param[i], f'Unused {conf_names[i]} parameters')) for i in range(2)]
    if args.seed==-1:
        args.seed = random.randint(1,int(2**31-1))

    if conf['use_gpu']:
        conf['device'] = 'cuda'

    if conf['training_device'] == 'cuda':
        if not torch.cuda.is_available():
            logger.info('CUDA is not available, using CPU for training instead.')
            conf['training_device'] = 'cpu'

    delattr(args, 'env_config_file')
    delattr(args, 'config_file')
    [save_yaml(c, f'{conf["save_dir"]}/{name}') for c, name in zip([conf, env_conf], ['conf','env_conf'])]
    return [Dotdict(x) for x in [vars(args), conf, env_conf]] + [conf['trainer']]

