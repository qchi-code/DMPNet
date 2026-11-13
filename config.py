import glob
import itertools
import os

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('DMPNet', save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds

for source_file in itertools.chain.from_iterable(
    glob.glob(f'{folder}/*.py') for folder in ['.', './dataloaders', './models']
):
    ex.add_source_file(source_file)


@ex.config
def cfg():
    seed = 2025
    gpu_id = 0
    num_workers = 2
    mode = 'train'
    dataset = 'CHAOST2'
    setting = 1
    exclude_label = None
    n_sv = 1000 if dataset == 'CMR' else 5000
    min_size = 200
    use_gt = False
    eval_fold = 0
    fold_protocol = 'legacy'
    test_label = [1, 2, 3] if dataset == 'CMR' else ([1, 2, 3, 6] if dataset == 'SABS' else [1, 2, 3, 4])
    supp_idx = 0
    n_part = 3
    image_size = 256

    n_steps = 40000
    batch_size = 1
    n_shot = 1
    n_way = 1
    n_query = 1
    max_iters_per_load = 1000
    print_interval = 1000
    save_snapshot_every = 10000

    ignore_label = 255
    lambda_psr = 0.5
    lambda_dice = 1.0
    bootstrap_max = 0.6
    bootstrap_warmup = 20000
    class_weights = [0.1, 1.0]
    grad_clip = 1.0
    lr_step_every = 1000
    lr_step_gamma = 0.98

    pretrained_weights = './checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth'
    use_mb_head = False
    legacy_extra_scales = False
    legacy_uot_row_normalize = False
    sinkhorn_iters = 7
    sinkhorn_eps = 0.15
    rw_iters = 6
    reload_model_path = None

    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True,
        'comp_lr': 2e-4,
        'ot_lr': 2e-4,
        'rw_lr': 2e-4,
        'gate_lr': 1e-4,
    }
    exp_str = '_'.join([mode, dataset, f'setting{setting}', f'cv{eval_fold}'])
    path = {
        'log_dir': './runs',
        'CHAOST2': {'data_dir': './data/CHAOST2'},
        'SABS': {'data_dir': './data/SABS'},
        'CMR': {'data_dir': './data/CMR'},
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    if config['dataset'] == 'CMR' and config['setting'] != 1:
        raise ValueError('CMR supports Setting 1 only.')
    if config['setting'] not in (1, 2):
        raise ValueError('setting must be 1 or 2')
    if config['mode'] == 'train':
        config['exclude_label'] = config['test_label'] if config['setting'] == 2 else None
    exp_name = f'{ex.path}_{config["exp_str"]}'
    ex.observers.append(FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name)))
    return config
