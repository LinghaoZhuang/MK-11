# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    # 检查模型类型，为不同架构提供不同的处理
    if hasattr(model, 'blocks'):  # Vision Transformer
        num_layers = len(model.blocks) + 1
        get_layer_id = get_layer_id_for_vit
    elif hasattr(model, 'model'):  # CNN 模型包装器
        if hasattr(model.model, 'stages'):  # ConvNeXt模型
            # ConvNeXt模型有4个stage + stem + head
            num_layers = 6
            get_layer_id = lambda name, num_layers: get_layer_id_for_convnext(name, num_layers, model)
        elif hasattr(model.model, 'layer4'):  # ResNet
            # ResNet通常有4个主要层 + stem + head
            num_layers = 6
            get_layer_id = lambda name, num_layers: get_layer_id_for_resnet(name, num_layers, model)
        elif hasattr(model.model, 'features'):  # VGG, MobileNet, EfficientNet等
            # 为简单起见，我们为这些模型使用统一的层次化策略
            num_layers = 5  # features + classifier
            get_layer_id = lambda name, num_layers: get_layer_id_for_features_classifier(name, num_layers, model)
        else:
            # 对于其他模型，不应用层级学习率衰减
            print("WARNING: Model architecture not recognized for layer-wise lr decay. Using single parameter group.")
            return [{'params': model.parameters(), 'weight_decay': weight_decay}]
    else:
        # 对于不支持的模型，使用单一参数组
        print("WARNING: Model architecture not recognized for layer-wise lr decay. Using single parameter group.")
        return [{'params': model.parameters(), 'weight_decay': weight_decay}]

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id for Vision Transformer
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def get_layer_id_for_resnet(name, num_layers, model):
    """
    Assign a parameter with its layer id for ResNet
    """
    if name.startswith('model.conv1') or name.startswith('model.bn1'):
        return 0  # stem
    elif name.startswith('model.layer1'):
        return 1
    elif name.startswith('model.layer2'):
        return 2
    elif name.startswith('model.layer3'):
        return 3
    elif name.startswith('model.layer4'):
        return 4
    elif name.startswith('model.fc') or name.startswith('model.classifier'):
        return 5  # head
    else:
        return num_layers


def get_layer_id_for_features_classifier(name, num_layers, model):
    """
    Assign a parameter with its layer id for models with features and classifier
    """
    if name.startswith('model.features'):
        # 根据深度对features内的层进行分组
        parts = name.split('.')
        if len(parts) >= 4 and parts[2].isdigit():
            layer_idx = int(parts[2])
            # 将features分为4组
            if hasattr(model.model, 'features') and len(model.model.features) > 0:
                features_len = len(model.model.features)
                if layer_idx < features_len // 4:
                    return 1
                elif layer_idx < features_len // 2:
                    return 2
                elif layer_idx < 3 * features_len // 4:
                    return 3
                else:
                    return 4
        return 1  # 默认features的前部分
    elif name.startswith('model.classifier'):
        return 5  # classifier
    else:
        return 0  # 其他部分


def get_layer_id_for_convnext(name, num_layers, model):
    """
    Assign a parameter with its layer id for ConvNeXt
    """
    if name.startswith('model.stem'):
        return 0  # stem
    elif name.startswith('model.stages'):
        parts = name.split('.')
        if len(parts) >= 3 and parts[2].isdigit():
            stage_idx = int(parts[2])
            # ConvNeXt有4个stage
            return stage_idx + 1  # stage_idx在0-3之间，对应返回1-4
        return 1  # 默认归为第一个stage
    elif name.startswith('model.head'):
        return 5  # head
    else:
        return 0  # 其他部分