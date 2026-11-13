#!/usr/bin/env python
import os
import random
import logging
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from models.fewshot import FewShotSeg
from dataloaders.datasets import TrainDataset as TrainDataset
from utils import *
from config import ex


def soft_dice_loss(prob_fg, target, ignore_index=255, smooth=1.0):
    if target.dim() == 3:
        t = target
    else:
        t = target.squeeze(1)
    valid = (t != ignore_index)
    y = (t == 1).float()
    p = prob_fg.float()
    p = p * valid.float()
    y = y * valid.float()
    inter = (p * y).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + y.sum(dim=(1, 2))
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def soft_bootstrap_ce(logits, target, beta, ignore_index=255, class_weight=None):
    B, C, H, W = logits.shape
    t = target
    valid = (t != ignore_index)
    if valid.sum() == 0:
        return logits.sum() * 0.0

    with torch.no_grad():
        prob = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(prob)
        onehot.scatter_(1, t.clamp(0, C - 1).unsqueeze(1), 1.0)
        onehot = onehot * valid.unsqueeze(1).float()
        y_soft = (1.0 - float(beta)) * onehot + float(beta) * (prob.detach() * valid.unsqueeze(1).float())

    logp = F.log_softmax(logits, dim=1)
    loss_map = -(y_soft * logp).sum(dim=1)

    if class_weight is not None:
        w = class_weight.to(logits.device)[t.clamp(0, C - 1)]
        loss_map = loss_map * w

    return loss_map[valid].mean()


def _build_optimizer(model, optim_cfg):
    base_lr = float(optim_cfg.get("lr", 1e-3))
    momentum = float(optim_cfg.get("momentum", 0.9))
    weight_decay = float(optim_cfg.get("weight_decay", 5e-4))
    nesterov = bool(optim_cfg.get("nesterov", True))
    dampening = float(optim_cfg.get("dampening", 0.0))

    comp_lr = float(optim_cfg.get("comp_lr", base_lr * 0.2))
    ot_lr = float(optim_cfg.get("ot_lr", base_lr * 0.2))
    rw_lr = float(optim_cfg.get("rw_lr", base_lr * 0.2))
    gate_lr = float(optim_cfg.get("gate_lr", base_lr * 0.1))

    comp_params, ot_params, rw_params, gate_params, base_params = [], [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("fg_components" in name) or ("bg_components" in name):
            comp_params.append(p)
        elif ("spatial_ot" in name) or ("ot_" in name):
            ot_params.append(p)
        elif ("rw_refine" in name) or ("rw_" in name):
            rw_params.append(p)
        elif ("gate" in name):
            gate_params.append(p)
        else:
            base_params.append(p)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr, "weight_decay": weight_decay})
    if comp_params:
        param_groups.append({"params": comp_params, "lr": comp_lr, "weight_decay": weight_decay})
    if ot_params:
        param_groups.append({"params": ot_params, "lr": ot_lr, "weight_decay": weight_decay})
    if rw_params:
        param_groups.append({"params": rw_params, "lr": rw_lr, "weight_decay": weight_decay})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})

    optimizer = torch.optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        dampening=dampening
    )
    return optimizer


def _build_scheduler(optimizer, n_steps, step_every=5000, gamma=0.95):
    step_every = int(step_every)
    milestones = list(range(step_every, int(n_steps), step_every))
    if len(milestones) == 0:
        milestones = [int(n_steps * 2 / 3)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(gamma))
    return scheduler, milestones


def _rand_uniform_like(x, a, b):
    return (b - a) * torch.rand((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype) + a


def _strong_augment_intensity(x):
    if x is None:
        return x
    y = x

    if torch.rand(1, device=y.device).item() < 0.9:
        gamma = _rand_uniform_like(y, 0.7, 1.5)
        y = torch.sign(y) * (torch.abs(y).clamp_min(1e-6) ** gamma)

    if torch.rand(1, device=y.device).item() < 0.9:
        contrast = _rand_uniform_like(y, 0.75, 1.25)
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * contrast + mean

    if torch.rand(1, device=y.device).item() < 0.9:
        bright = _rand_uniform_like(y, -0.10, 0.10)
        y = y + bright

    if torch.rand(1, device=y.device).item() < 0.7:
        sigma = _rand_uniform_like(y, 0.0, 0.05)
        y = y + torch.randn_like(y) * sigma

    if torch.rand(1, device=y.device).item() < 0.5:
        k = 3 if torch.rand(1, device=y.device).item() < 0.7 else 5
        y = F.avg_pool2d(y, kernel_size=k, stride=1, padding=k // 2)

    return y


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', 'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    if _config['seed'] is not None:
        random.seed(_config['seed'])
        np.random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = False

    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device(f'cuda:{_config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    _log.info('Create model...')
    model = FewShotSeg(
        pretrained_weights=_config['pretrained_weights'],
        use_mb_head=_config['use_mb_head'],
        sinkhorn_iters=_config['sinkhorn_iters'],
        sinkhorn_eps=_config['sinkhorn_eps'],
        rw_iters=_config['rw_iters'],
        legacy_extra_scales=_config['legacy_extra_scales'],
        legacy_uot_row_normalize=_config['legacy_uot_row_normalize'],
    ).to(device)
    if _config.get('reload_model_path'):
        checkpoint = torch.load(_config['reload_model_path'], map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)

    model.train()

    _log.info('Set optimizer...')
    optimizer = _build_optimizer(model, _config['optim'])

    sched_every = int(_config.get("lr_step_every", 5000))
    scheduler, milestones = _build_scheduler(
        optimizer,
        n_steps=int(_config['n_steps']),
        step_every=sched_every,
        gamma=float(_config['lr_step_gamma'])
    )
    _log.info(f'LR milestones: {milestones}, gamma: {_config["lr_step_gamma"]}, step_every: {sched_every}')

    my_weight = (torch.tensor(_config['class_weights'], dtype=torch.float, device=device)
                 if _config['class_weights'] is not None else None)

    lambda_prc = float(_config['lambda_psr'])
    lambda_dice = float(_config.get("lambda_dice", 1.0))

    boot_beta_max = float(_config['bootstrap_max'])
    boot_warmup_steps = int(_config['bootstrap_warmup'])

    grad_clip = float(_config.get("grad_clip", 1.0))

    _log.info('Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
        'fold_protocol': _config['fold_protocol'],
        'image_size': _config['image_size'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    n_sub_epochs = math.ceil(_config['n_steps'] / _config['max_iters_per_load'])
    log_loss = {
        'total_loss': 0.0,
        'ce_loss': 0.0,
        'dice_loss': 0.0,
        'prc_loss': 0.0,
    }

    i_iter = 0
    _log.info('Start training...')

    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')

        if hasattr(model, "set_epoch"):
            model.set_epoch(sub_epoch)

        for _, sample in enumerate(train_loader):
            support_images = [[shot.float().to(device, non_blocking=True) for shot in way] for way in sample['support_images']]
            support_fg_mask = [[shot.float().to(device, non_blocking=True) for shot in way] for way in sample['support_fg_labels']]
            query_images = [query_image.float().to(device, non_blocking=True) for query_image in sample['query_images']]
            query_labels = torch.cat(
                [query_label.long().to(device, non_blocking=True) for query_label in sample['query_labels']], dim=0
            )

            query_images_aug = [_strong_augment_intensity(query_images[0])]
            query_logits, prc_loss = model(support_images, support_fg_mask, query_images_aug, train=True)

            beta = min(float(i_iter) / float(max(boot_warmup_steps, 1)), 1.0) * boot_beta_max
            ce_loss = soft_bootstrap_ce(
                query_logits, query_labels,
                beta=beta,
                ignore_index=255,
                class_weight=my_weight
            )

            prob = torch.softmax(query_logits, dim=1)
            prob_fg = prob[:, 1, :, :]
            dice_loss = soft_dice_loss(prob_fg, query_labels, ignore_index=255)

            loss = ce_loss + lambda_dice * dice_loss + lambda_prc * prc_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            ce_val = float(ce_loss.detach().cpu().item())
            dice_val = float(dice_loss.detach().cpu().item())
            prc_val = float(prc_loss.detach().cpu().item())
            total_val = float(loss.detach().cpu().item())

            _run.log_scalar('total_loss', total_val)
            _run.log_scalar('ce_loss', ce_val)
            _run.log_scalar('dice_loss', dice_val)
            _run.log_scalar('prc_loss', prc_val)

            log_loss['total_loss'] += total_val
            log_loss['ce_loss'] += ce_val
            log_loss['dice_loss'] += dice_val
            log_loss['prc_loss'] += prc_val

            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                ce_avg = log_loss['ce_loss'] / _config['print_interval']
                dice_avg = log_loss['dice_loss'] / _config['print_interval']
                prc_avg = log_loss['prc_loss'] / _config['print_interval']

                log_loss = {
                    'total_loss': 0.0,
                    'ce_loss': 0.0,
                    'dice_loss': 0.0,
                    'prc_loss': 0.0,
                }

                _log.info(
                    f'step {i_iter + 1}: total_loss: {total_loss}, '
                    f'ce_loss: {ce_avg}, dice_loss: {dice_avg}, prc_loss: {prc_avg}'
                )

            i_iter += 1
            if i_iter % int(_config['save_snapshot_every']) == 0 and _run.observers:
                torch.save(model.state_dict(), os.path.join(
                    _run.observers[0].dir, 'snapshots', f'{i_iter}.pth'))

            if i_iter >= int(_config['n_steps']):
                break
        if i_iter >= int(_config['n_steps']):
            break

    _log.info('End of training.')
    if _run.observers:
        torch.save(model.state_dict(), os.path.join(_run.observers[0].dir, 'snapshots', 'final.pth'))
    return 1
