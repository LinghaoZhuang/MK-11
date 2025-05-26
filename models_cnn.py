# Copyright (c) 2023
# All rights reserved.

# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.resnet
import timm.models.vgg
import timm.models.mobilenetv3
import timm.models.efficientnet
import timm.models.regnet
import timm.models.densenet
import timm.models.convnext


class ResNetWrapper(nn.Module):
    """ ResNet with optional global average pooling
    """
    def __init__(self, model, num_classes=1000, global_pool=True):
        super().__init__()
        self.model = model
        self.global_pool = global_pool
        
        # Replace the last FC layer to match num_classes
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        
        self.no_weight_decay_list = []  # Required for compatibility with training script

    def forward(self, x):
        return self.model(x)
        
    def no_weight_decay(self):
        return self.no_weight_decay_list


class VGGWrapper(nn.Module):
    """ VGG with optional modifications
    """
    def __init__(self, model, num_classes=1000, global_pool=True):
        super().__init__()
        self.model = model
        self.global_pool = global_pool
        
        # Replace the last classifier layer to match num_classes
        if hasattr(self.model, 'head'):
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, num_classes)
        
        self.no_weight_decay_list = []  # Required for compatibility with training script

    def forward(self, x):
        return self.model(x)
        
    def no_weight_decay(self):
        return self.no_weight_decay_list


class MobileNetWrapper(nn.Module):
    """ MobileNet with optional modifications
    """
    def __init__(self, model, num_classes=1000, global_pool=True):
        super().__init__()
        self.model = model
        self.global_pool = global_pool
        
        # Replace the last classifier layer to match num_classes
        if hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, num_classes)
        
        self.no_weight_decay_list = []  # Required for compatibility with training script

    def forward(self, x):
        return self.model(x)
        
    def no_weight_decay(self):
        return self.no_weight_decay_list


class EfficientNetWrapper(nn.Module):
    """ EfficientNet with optional modifications
    """
    def __init__(self, model, num_classes=1000, global_pool=True):
        super().__init__()
        self.model = model
        self.global_pool = global_pool
        
        # Replace the last classifier layer to match num_classes
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        
        self.no_weight_decay_list = []  # Required for compatibility with training script

    def forward(self, x):
        return self.model(x)
        
    def no_weight_decay(self):
        return self.no_weight_decay_list


class ConvNeXtWrapper(nn.Module):
    """ ConvNeXt with optional modifications
    """
    def __init__(self, model, num_classes=1000, global_pool=True):
        super().__init__()
        self.model = model
        self.global_pool = global_pool
        
        # Replace the last classifier layer to match num_classes
        if hasattr(self.model, 'head') and isinstance(self.model.head, nn.Sequential):
            if hasattr(self.model.head, 'fc'):
                in_features = self.model.head.fc.in_features
                self.model.head.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'head'):
            if hasattr(self.model.head, 'classifier'):
                in_features = self.model.head.classifier.in_features
                self.model.head.classifier = nn.Linear(in_features, num_classes)
        
        self.no_weight_decay_list = []  # Required for compatibility with training script

    def forward(self, x):
        return self.model(x)
        
    def no_weight_decay(self):
        return self.no_weight_decay_list


# ResNet family
def resnet18(pretrained=False, **kwargs):
    base_model = timm.models.resnet.resnet18(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    base_model = timm.models.resnet.resnet34(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def resnet50(pretrained=False, **kwargs):
    base_model = timm.models.resnet.resnet50(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def resnet101(pretrained=False, **kwargs):
    base_model = timm.models.resnet.resnet101(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def resnet152(pretrained=False, **kwargs):
    base_model = timm.models.resnet.resnet152(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

# VGG family
def vgg16(pretrained=False, **kwargs):
    base_model = timm.models.vgg.vgg16(pretrained=pretrained)
    model = VGGWrapper(base_model, **kwargs)
    return model

def vgg19(pretrained=False, **kwargs):
    base_model = timm.models.vgg.vgg19(pretrained=pretrained)
    model = VGGWrapper(base_model, **kwargs)
    return model

# MobileNet family
def mobilenet_v3_small(pretrained=False, **kwargs):
    base_model = timm.models.mobilenetv3.mobilenetv3_small_100(pretrained=pretrained)
    model = MobileNetWrapper(base_model, **kwargs)
    return model

def mobilenet_v3_large(pretrained=False, **kwargs):
    base_model = timm.models.mobilenetv3.mobilenetv3_large_100(pretrained=pretrained)
    model = MobileNetWrapper(base_model, **kwargs)
    return model

# EfficientNet family
def efficientnet_b0(pretrained=False, **kwargs):
    base_model = timm.models.efficientnet.efficientnet_b0(pretrained=pretrained)
    model = EfficientNetWrapper(base_model, **kwargs)
    return model

def efficientnet_b1(pretrained=False, **kwargs):
    base_model = timm.models.efficientnet.efficientnet_b1(pretrained=pretrained)
    model = EfficientNetWrapper(base_model, **kwargs)
    return model

def efficientnet_b2(pretrained=False, **kwargs):
    base_model = timm.models.efficientnet.efficientnet_b2(pretrained=pretrained)
    model = EfficientNetWrapper(base_model, **kwargs)
    return model

def efficientnet_b3(pretrained=False, **kwargs):
    base_model = timm.models.efficientnet.efficientnet_b3(pretrained=pretrained)
    model = EfficientNetWrapper(base_model, **kwargs)
    return model

# RegNet family
def regnet_y_400mf(pretrained=False, **kwargs):
    base_model = timm.models.regnet.regnetx_004(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def regnet_y_8gf(pretrained=False, **kwargs):
    base_model = timm.models.regnet.regnetx_080(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

# DenseNet family
def densenet121(pretrained=False, **kwargs):
    base_model = timm.models.densenet.densenet121(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def densenet169(pretrained=False, **kwargs):
    base_model = timm.models.densenet.densenet169(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

def densenet201(pretrained=False, **kwargs):
    base_model = timm.models.densenet.densenet201(pretrained=pretrained)
    model = ResNetWrapper(base_model, **kwargs)
    return model

# ConvNeXt family
def convnext_tiny(pretrained=False, **kwargs):
    base_model = timm.models.convnext.convnext_tiny(pretrained=pretrained)
    model = ConvNeXtWrapper(base_model, **kwargs)
    return model

def convnext_small(pretrained=False, **kwargs):
    base_model = timm.models.convnext.convnext_small(pretrained=pretrained)
    model = ConvNeXtWrapper(base_model, **kwargs)
    return model

def convnext_base(pretrained=False, **kwargs):
    base_model = timm.models.convnext.convnext_base(pretrained=pretrained)
    model = ConvNeXtWrapper(base_model, **kwargs)
    return model

def convnext_large(pretrained=False, **kwargs):
    base_model = timm.models.convnext.convnext_large(pretrained=pretrained)
    model = ConvNeXtWrapper(base_model, **kwargs)
    return model 