import yaml
import argparse

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_args_with_config(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default=config['train'].get('logdir', 'runs/'))
    parser.add_argument('--seed', type=int, default=config['train'].get('seed', 42))
    parser.add_argument('--lr', type=float, default=config['train']['lr'])
    parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
    parser.add_argument('--total_steps', type=int, default=config['train']['total_steps'])
    args = parser.parse_args()

    config['train']['logdir'] = args.logdir
    config['train']['seed'] = args.seed
    config['train']['lr'] = args.lr
    config['train']['batch_size'] = args.batch_size
    config['train']['total_steps'] = args.total_steps
    return config