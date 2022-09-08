import sys
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from torch.nn.parameter import Parameter

_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

_cfg_cifar10 = {
    'url': '',
    'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2471, 0.2435, 0.261), 'classifier': 'head'
}

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, dilation=1, patch_size=7, n_classes=1000, apply_kernel_sharing=True):
    forward_block = []
    dw_init = nn.Conv2d(dim, dim, kernel_size, dilation=dilation, groups=dim, padding=dilation * (kernel_size - 1) // 2)
    pw_init = nn.Conv2d(dim, dim, kernel_size=1)

    pw_w = Parameter(pw_init.weight.clone().detach())
    dw_w = Parameter(dw_init.weight.clone().detach())
    pw_b = Parameter(pw_init.bias.clone().detach())
    dw_b = Parameter(dw_init.bias.clone().detach())

    for i in range(depth):
        dw_temp = nn.Conv2d(dim, dim, kernel_size, dilation=dilation, groups=dim, padding=dilation * (kernel_size - 1) // 2)
        pw_temp = nn.Conv2d(dim, dim, kernel_size=1)

        if apply_kernel_sharing:
            pw_temp.weight = pw_w
            dw_temp.weight = dw_w
            pw_temp.bias = pw_b
            dw_temp.bias = dw_b

        temp = nn.Sequential(
            Residual(nn.Sequential(
                dw_temp,
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            pw_temp,
            nn.GELU(),
            nn.BatchNorm2d(dim))
        forward_block.append(temp)
    
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *forward_block,
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


@register_model
def convmixer_256_16_shared(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=9,  patch_size=1, n_classes=kwargs.get("num_classes"), apply_kernel_sharing=True)
    model.default_cfg = _cfg    
    return model

@register_model
def convmixer_256_16(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=9,  patch_size=1, n_classes=kwargs.get("num_classes"), apply_kernel_sharing=False)
    model.default_cfg = _cfg    
    return model