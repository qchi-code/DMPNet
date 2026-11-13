#!/usr/bin/env python3
import csv
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import ex
from dataloaders.dataset_specifics import get_label_names
from dataloaders.datasets import TestDataset
from models.fewshot import FewShotSeg
from utils import Scores


@ex.automain
def main(_run, _config, _log):
    if not _config.get('reload_model_path'):
        raise ValueError('Specify reload_model_path for evaluation. See README.md.')
    if not os.path.isfile(_config['reload_model_path']):
        raise FileNotFoundError(_config['reload_model_path'])
    if _run.observers:
        run_dir = _run.observers[0].dir
        os.makedirs(run_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(run_dir, 'logger.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        _log.handlers.append(handler)
    else:
        run_dir = '.'
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        np.random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
    cudnn.enabled = True
    device = torch.device(f'cuda:{_config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = FewShotSeg(
        pretrained_weights=None,
        use_mb_head=_config['use_mb_head'],
        sinkhorn_iters=_config['sinkhorn_iters'],
        sinkhorn_eps=_config['sinkhorn_eps'],
        rw_iters=_config['rw_iters'],
        legacy_extra_scales=_config['legacy_extra_scales'],
        legacy_uot_row_normalize=_config['legacy_uot_row_normalize'],
    )
    state = torch.load(_config['reload_model_path'], map_location='cpu', weights_only=False)
    state = state.get('model_state_dict', state.get('state_dict', state))
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'eval_fold': _config['eval_fold'],
        'supp_idx': _config['supp_idx'],
        'fold_protocol': _config['fold_protocol'],
        'image_size': _config['image_size'],
    }
    dataset = TestDataset(data_config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=_config['num_workers'])
    labels = get_label_names(_config['dataset'])
    rows = []
    class_scores = {}
    _log.info('Evaluate DMPNet on %s, fold=%s, setting=%s, chunks=%s',
              _config['dataset'], _config['eval_fold'], _config['setting'], _config['n_part'])
    with torch.no_grad():
        for label_val, label_name in labels.items():
            if label_val == 0 or label_val not in _config['test_label']:
                continue
            support_sample = dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])
            dataset.label = label_val
            support_image = [item.unsqueeze(0).to(device, dtype=torch.float32)
                             for item in support_sample['image']]
            support_mask = [item.unsqueeze(0).to(device, dtype=torch.float32)
                            for item in support_sample['label']]
            scores = Scores()
            for sample in loader:
                query_volume = sample['image'][0]
                truth = sample['label'][0].long()
                if truth.shape[0] == 0:
                    continue
                pred = torch.zeros_like(truth)
                boundaries = np.linspace(0, query_volume.shape[0], _config['n_part']+1, dtype=int)
                for i in range(_config['n_part']):
                    for z in range(boundaries[i], boundaries[i+1]):
                        image = query_volume[z:z+1].to(device, dtype=torch.float32)
                        probability = model([[support_image[i]]], [[support_mask[i]]], [image], train=False)
                        pred[z] = probability.argmax(dim=1)[0].cpu()
                spacing = sample['spacing'][0].numpy()
                dice, iou, hd95 = scores.record(pred, truth, spacing=spacing)
                case = os.path.basename(sample['id'][0])
                rows.append((label_name, case, dice, iou, hd95))
                _log.info('%s | %s | Dice %.4f | IoU %.4f | HD95 %.3f mm',
                          label_name, case, dice, iou, hd95)
            if not scores.patient_dice:
                _log.warning('No query volumes with foreground for class %s', label_name)
                continue
            class_scores[label_name] = (
                float(np.mean(scores.patient_dice)),
                float(np.mean(scores.patient_iou)),
                float(np.mean(scores.patient_hd95)),
            )
            _log.info('%s mean Dice=%.4f HD95=%.3f mm', label_name,
                      class_scores[label_name][0], class_scores[label_name][2])
    if not class_scores:
        raise ValueError('No valid evaluation results')
    summary = (
        np.mean([entry[0] for entry in class_scores.values()]),
        np.mean([entry[1] for entry in class_scores.values()]),
        np.mean([entry[2] for entry in class_scores.values()]),
    )
    out_path = os.path.join(run_dir, 'metrics.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['class', 'case', 'dice', 'iou', 'hd95_mm'])
        writer.writerows(rows)
        for name, metrics in class_scores.items():
            writer.writerow([name, 'MEAN', *metrics])
        writer.writerow(['ALL', 'MEAN', *summary])
    _log.info('Overall class-macro Dice=%.4f, IoU=%.4f, HD95=%.3f mm', *summary)
    _log.info('Metrics saved to %s', out_path)
    return float(summary[0])
