#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.segmentation import slic

SUBDIR = {
    'CHAOST2': 'chaos_MR_T2_normalized',
    'SABS': 'sabs_CT_normalized',
    'CMR': 'cmr_MR_normalized',
}


def main():
    parser = argparse.ArgumentParser(description='Generate 3D SLIC pseudo-label volumes for episodic training.')
    parser.add_argument('--dataset', required=True, choices=sorted(SUBDIR))
    parser.add_argument('--data-dir', required=True, help='Dataset root containing the normalized image subfolder')
    parser.add_argument('--n-segments', type=int, default=None)
    parser.add_argument('--compactness', type=float, default=0.05)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    n_segments = args.n_segments or (1000 if args.dataset == 'CMR' else 5000)
    root = Path(args.data_dir).expanduser()
    source = root / SUBDIR[args.dataset]
    destination = root / f'supervoxels_{n_segments}'
    files = sorted(source.glob('image_*.nii.gz'))
    if not files:
        raise FileNotFoundError(f'No image_*.nii.gz volumes in {source}')
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        match = re.search(r'image_(\d+)\.nii\.gz$', path.name)
        if match is None:
            continue
        target = destination / f'super_{match.group(1)}.nii.gz'
        if target.exists() and not args.overwrite:
            print(f'Skip: {target}')
            continue
        volume = sitk.ReadImage(str(path))
        image = sitk.GetArrayFromImage(volume).astype(np.float32)
        image = (image - image.mean()) / max(float(image.std()), 1e-6)
        segmentation = slic(image, n_segments=n_segments, compactness=args.compactness,
                            sigma=args.sigma, start_label=1, channel_axis=None,
                            enforce_connectivity=True).astype(np.int32)
        generated = sitk.GetImageFromArray(segmentation)
        generated.CopyInformation(volume)
        sitk.WriteImage(generated, str(target), useCompression=True)
        print(f'{path.name} -> {target.name}, components={len(np.unique(segmentation))}')


if __name__ == '__main__':
    main()
