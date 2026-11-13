from pathlib import Path

import torch
import torch.nn as nn
import torchvision


class _ResNetEncoder(nn.Module):
    def __init__(self, depth=101, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__()
        if depth == 101:
            net = torchvision.models.resnet101(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)
        elif depth == 50:
            net = torchvision.models.resnet50(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            raise ValueError(f'Unsupported ResNet depth: {depth}')
        self.backbone = nn.ModuleDict(dict(net.named_children()))
        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(1000, 1)
        self._init_weights()
        if pretrained_weights is not None:
            self._load_weights(pretrained_weights)

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _load_weights(self, source):
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f'Pretrained encoder weights not found: {path}. '
                    'Set pretrained_weights to the local MS-COCO DeeplabV3-ResNet101 checkpoint.'
                )
            state = torch.load(path, map_location='cpu', weights_only=False)
        elif isinstance(source, dict):
            state = source
        else:
            raise TypeError('pretrained_weights must be a checkpoint path, state dictionary or None')
        if 'state_dict' in state:
            state = state['state_dict']
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        target = self.state_dict()
        matches = {key: value for key, value in state.items()
                   if key in target and tuple(target[key].shape) == tuple(value.shape)}
        if len(matches) < 100:
            raise ValueError(
                f'Only {len(matches)} encoder tensors matched the supplied weights. '
                'Expected a matching DeeplabV3/ResNet checkpoint with backbone.* keys.'
            )
        target.update(matches)
        self.load_state_dict(target)
        print(f'Initialized encoder with {len(matches)} tensors from pretrained checkpoint.')

    def forward(self, x):
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)
        x = self.backbone['maxpool'](x)
        x = self.backbone['layer1'](x)
        x = self.backbone['layer2'](x)
        x = self.backbone['layer3'](x)
        feature = self.reduce1(x)
        x = self.backbone['layer4'](x)
        t = self.backbone['avgpool'](x).flatten(1)
        t = self.reduce1d(self.backbone['fc'](t))
        return feature, t


class Res101Encoder(_ResNetEncoder):
    def __init__(self, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__(101, replace_stride_with_dilation, pretrained_weights)


class Res50Encoder(_ResNetEncoder):
    def __init__(self, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__(50, replace_stride_with_dilation, pretrained_weights)
