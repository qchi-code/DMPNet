"""
Experiment configuration file
"""
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from utils import *

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment("FSMIS_four", save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './utils']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


@ex.config
def cfg():
    seed = 2025
    gpu_id = 0
    num_workers = 2
    mode = 'train'

    dataset = 'CHAOST2'
    exclude_label = None
    if dataset == 'CMR':
        n_sv = 1000
    else:
        n_sv = 5000
    min_size = 200
    max_slices = 3
    use_gt = False
    eval_fold = 0
    test_label = [1, 2, 3, 4]
    supp_idx = 0
    n_part = 3

    n_steps = 30000
    batch_size = 1
    n_shot = 1
    n_way = 1
    n_query = 1

    max_iters_per_load = 1000
    print_interval = 1000
    save_snapshot_every = 10000

    ignore_label = 255
    bg_wt = 0.1
    t_loss_scaler = 0.0

    reload_model_path = None

    lambda_prc = 0.1
    lambda_dice = 1.0

    grad_clip = 1.0

    lr_step_every = 5000
    lr_step_gamma = 0.95

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': True,
        'comp_lr': 2e-4,
        'mamba_lr': 2e-4,
        'gate_lr': 1e-4,
    }

    exp_str = '_'.join(
        [mode] + [dataset] + [f'cv{eval_fold}']
    )

    path = {
        'log_dir': './runs',
        'CHAOST2': {'data_dir': '/home/ac/data/cq/RPT-main/data/CHAOST2'},
        'SABS': {'data_dir': '/home/ac/data/cq/RPT-main/data/SABS'},
        'CMR': {'data_dir': '/home/ac/data/cq/RPT-main/data/CMR'},
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
