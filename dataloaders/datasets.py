import glob
import os
import random
import re

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx

from . import image_transforms as myit
from .dataset_specifics import get_folds


DATA_SUBDIRS = {
    'CHAOST2': 'chaos_MR_T2_normalized',
    'SABS': 'sabs_CT_normalized',
    'CMR': 'cmr_MR_normalized',
}


def _case_id(path):
    found = re.search(r'_(\d+)\.nii(?:\.gz)?$', os.path.basename(path))
    if not found:
        raise ValueError(f'Could not parse patient ID: {path}')
    return int(found.group(1))


def _paths(root, dataset, prefix):
    directory = os.path.join(root, DATA_SUBDIRS[dataset])
    return sorted(glob.glob(os.path.join(directory, f'{prefix}_*.nii.gz')), key=_case_id)


def _pairs(root, dataset):
    images = _paths(root, dataset, 'image')
    labels = _paths(root, dataset, 'label')
    if not images:
        raise FileNotFoundError(f'No image_*.nii.gz files found in {root}/{DATA_SUBDIRS[dataset]}')
    label_lookup = {_case_id(path): path for path in labels}
    pairs = []
    for path in images:
        case_id = _case_id(path)
        if case_id not in label_lookup:
            raise FileNotFoundError(f'Missing label_{case_id}.nii.gz for {path}')
        pairs.append((path, label_lookup[case_id]))
    return pairs


def _normalize(image):
    image = image.astype(np.float32, copy=False)
    return (image - image.mean()) / max(float(image.std()), 1e-6)


def _remap_label(label):
    label = label.copy()
    label[label == 200] = 1
    label[label == 500] = 2
    label[label == 600] = 3
    return label


def _resize_volume(volume, size, is_label=False):
    if volume.shape[-2:] == (size, size):
        return volume
    interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    return np.stack([cv2.resize(frame, (size, size), interpolation=interpolation)
                     for frame in volume], axis=0)


class TestDataset(Dataset):
    def __init__(self, args):
        self.dataset = args['dataset']
        self.size = int(args.get('image_size', 256))
        self.fold_protocol = args.get('fold_protocol', 'legacy')
        pairs = _pairs(args['data_dir'], self.dataset)
        folds = get_folds(self.dataset, self.fold_protocol)
        selected = [pair for idx, pair in enumerate(pairs) if idx in folds[args['eval_fold']]]
        support_idx = int(args['supp_idx'])
        if not 0 <= support_idx < len(selected):
            raise IndexError(f'supp_idx={support_idx} is invalid for {len(selected)} patients in fold')
        self.support_pair = selected.pop(support_idx)
        self.pairs = selected
        self.label = None

    def __len__(self):
        return len(self.pairs)

    def _load(self, pair, target):
        img_sitk = sitk.ReadImage(pair[0])
        img_raw = sitk.GetArrayFromImage(img_sitk)
        label = _remap_label(sitk.GetArrayFromImage(sitk.ReadImage(pair[1])))
        spacing = np.array(img_sitk.GetSpacing()[::-1], dtype=np.float32)
        z, h, w = img_raw.shape
        spacing[-2] *= h / self.size
        spacing[-1] *= w / self.size
        img = _resize_volume(_normalize(img_raw), self.size)
        label = _resize_volume((label == target).astype(np.uint8), self.size, is_label=True)
        img = np.repeat(img[:, None], 3, axis=1)
        return img, label, spacing

    def __getitem__(self, index):
        if self.label is None:
            raise ValueError('Set test_dataset.label before querying the dataset.')
        pair = self.pairs[index]
        image, label, spacing = self._load(pair, self.label)
        roi = label.sum(axis=(1, 2)) > 0
        return {
            'id': pair[0],
            'image': torch.from_numpy(image[roi].copy()),
            'label': torch.from_numpy(label[roi].copy()),
            'spacing': torch.from_numpy(spacing),
        }

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('label is required')
        image, mask, spacing = self._load(self.support_pair, label)
        if not all_slices:
            if N is None or N < 1:
                raise ValueError('Positive N is required')
            indices = np.flatnonzero(mask.sum(axis=(1, 2)) > 0)
            if len(indices) == 0:
                raise ValueError(f'No support foreground pixels for class {label} in {self.support_pair[0]}')
            boundaries = np.linspace(0, len(indices), int(N) + 1, dtype=int)
            selected = [indices[min((boundaries[j] + boundaries[j+1]) // 2, len(indices)-1)]
                        for j in range(int(N))]
            image, mask = image[selected], mask[selected]
        return {'image': torch.from_numpy(image.copy()),
                'label': torch.from_numpy(mask.copy()),
                'spacing': torch.from_numpy(spacing)}


class TrainDataset(Dataset):
    def __init__(self, args):
        self.n_shot = int(args['n_shot'])
        self.n_way = int(args['n_way'])
        self.n_query = int(args['n_query'])
        if self.n_way != 1 or self.n_shot != 1 or self.n_query != 1:
            raise ValueError('The published 1-way 1-shot training pipeline requires n_way=n_shot=n_query=1')
        self.max_iter = int(args['max_iter'])
        self.min_size = int(args['min_size'])
        self.use_gt = bool(args['use_gt'])
        self.test_label = set(args['test_label'])
        self.exclude_label = set(args['exclude_label'] or [])
        self.size = int(args.get('image_size', 256))
        dataset = args['dataset']
        pairs = _pairs(args['data_dir'], dataset)
        folds = get_folds(dataset, args.get('fold_protocol', 'legacy'))
        self.cases = []
        sv_root = os.path.join(args['data_dir'], f'supervoxels_{args["n_sv"]}')
        sv_lookup = {_case_id(path): path for path in glob.glob(os.path.join(sv_root, 'super_*.nii.gz'))}
        for index, pair in enumerate(pairs):
            if index in folds[args['eval_fold']]:
                continue
            case_id = _case_id(pair[0])
            if not self.use_gt and case_id not in sv_lookup:
                raise FileNotFoundError(
                    f'Missing 3D supervoxel pseudo-label for patient {case_id} under {sv_root}. '
                    'Run scripts/prepare_supervoxels.py or supply the original pseudo-labels.'
                )
            self.cases.append((pair, sv_lookup.get(case_id)))
        if not self.cases:
            raise ValueError('Training fold has no patients')
        self.images = []
        self.labels = []
        self.supervoxels = []
        for (image_path, label_path), super_path in self.cases:
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            gt = _remap_label(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))
            sv = sitk.GetArrayFromImage(sitk.ReadImage(super_path)) if not self.use_gt else gt
            if image.shape != gt.shape or image.shape != sv.shape:
                raise ValueError(f'Shape mismatch for {image_path}')
            self.images.append(_resize_volume(_normalize(image), self.size))
            self.labels.append(_resize_volume(gt, self.size, is_label=True))
            self.supervoxels.append(_resize_volume(sv.astype(np.int32), self.size, is_label=True))
        self.geometry = deftfx.Compose([
            myit.RandomAffine(5, (5, 5), 5, (0.9, 1.2), True, order=3),
            myit.ElasticTransform(10, 5),
        ])

    def __len__(self):
        return self.max_iter

    @staticmethod
    def _largest_connected_component(mask):
        from scipy.ndimage import label
        components, n = label(mask.astype(bool))
        if n == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        counts = np.bincount(components.reshape(-1))
        counts[0] = 0
        return (components == counts.argmax()).astype(np.uint8)

    @staticmethod
    def _gamma_transform(x):
        gamma = np.random.uniform(0.5, 1.5)
        low = x.min()
        span = max(float(x.max()-low), 1e-5)
        return span * np.power(np.clip((x-low)/span, 0, 1), gamma) + low

    def _geom_transform(self, images, masks):
        for i in range(len(images)):
            combined = np.concatenate([images[i].transpose(1, 2, 0), masks[i][..., None]], axis=2)
            transformed = self.geometry(combined)
            images[i] = transformed[..., :3].transpose(2, 0, 1)
            masks[i] = np.rint(transformed[..., -1]).clip(0, 1)
        return images, masks

    def __getitem__(self, index):
        num = self.n_shot + self.n_query
        for _ in range(200):
            patient = random.randrange(len(self.cases))
            image = self.images[patient]
            gt = self.labels[patient]
            classes = self.supervoxels[patient]
            valid_slices = np.ones(len(image), dtype=bool)
            for label_id in self.exclude_label:
                valid_slices &= ~(gt == label_id).any(axis=(1, 2))
            unique = np.unique(classes)
            unique = [int(c) for c in unique if c != 0 and (not self.use_gt or c not in self.test_label)]
            if not unique:
                continue
            category = random.choice(unique)
            binary = classes == category
            slice_size = binary.sum(axis=(1, 2))
            possible = []
            for start in range(len(image) - num + 1):
                ids = np.arange(start, start + num)
                if valid_slices[ids].all() and (slice_size[ids] >= self.min_size).all():
                    possible.append(start)
            if not possible:
                continue
            first = random.choice(possible)
            indices = np.arange(first, first + num)
            binary_slices = np.stack([self._largest_connected_component(binary[i]) for i in indices])
            if (binary_slices.sum(axis=(1, 2)) < self.min_size).any():
                continue
            if np.random.random() > 0.5:
                indices = indices[::-1]
                binary_slices = binary_slices[::-1]
            selected = image[indices]
            img = np.repeat(selected[:, None], 3, axis=1).astype(np.float32)
            sup_img, qry_img = img[:self.n_shot].copy(), img[self.n_shot:].copy()
            sup_lbl = binary_slices[:self.n_shot].copy().astype(np.float32)
            qry_lbl = binary_slices[self.n_shot:].copy().astype(np.float32)
            if np.random.random() > 0.5:
                qry_img = self._gamma_transform(qry_img)
            else:
                sup_img = self._gamma_transform(sup_img)
            if np.random.random() > 0.5:
                qry_img, qry_lbl = self._geom_transform(qry_img, qry_lbl)
            else:
                sup_img, sup_lbl = self._geom_transform(sup_img, sup_lbl)
            return {
                'support_images': np.ascontiguousarray(sup_img[None], dtype=np.float32),
                'support_fg_labels': np.ascontiguousarray(sup_lbl[None], dtype=np.float32),
                'query_images': np.ascontiguousarray(qry_img, dtype=np.float32),
                'query_labels': np.ascontiguousarray(qry_lbl, dtype=np.int64),
                'selected_class': int(category),
            }
        raise RuntimeError(
            'Could not form an episodic task after 200 attempts. Check supervoxel counts, '
            'min_size and Setting 2 exclusion labels.'
        )
