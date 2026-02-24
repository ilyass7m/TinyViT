# --------------------------------------------------------
# TinyViT Model Builder
# Copyright (c) 2022 Microsoft
# Extended to support ResNet teachers for distillation
# --------------------------------------------------------

from .tiny_vit import TinyViT


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'tiny_vit':
        M = config.MODEL.TINY_VIT
        model = TinyViT(img_size=config.DATA.IMG_SIZE,
                        in_chans=M.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dims=M.EMBED_DIMS,
                        depths=M.DEPTHS,
                        num_heads=M.NUM_HEADS,
                        window_sizes=M.WINDOW_SIZES,
                        mlp_ratio=M.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        mbconv_expand_ratio=M.MBCONV_EXPAND_RATIO,
                        local_conv_size=M.LOCAL_CONV_SIZE,
                        layer_lr_decay=config.TRAIN.LAYER_LR_DECAY,
                        )
    elif model_type == 'clip_vit_large14_224':
        from .clip import CLIP
        kwargs = {
            'embed_dim': 768, 'image_resolution': 224,
            'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14,
            "num_classes": config.MODEL.NUM_CLASSES,
        }
        model = CLIP(**kwargs)
    elif model_type in ['resnet50', 'resnet101', 'resnet152']:
        # ResNet models for teacher distillation
        # Use torchvision instead of timm - timm 0.4.12 lacks pretrained weights for resnet152
        import torch.nn as nn
        import torchvision.models as tv_models

        # Map model names to torchvision functions and weights
        resnet_map = {
            'resnet50': (tv_models.resnet50, 'IMAGENET1K_V1'),
            'resnet101': (tv_models.resnet101, 'IMAGENET1K_V1'),
            'resnet152': (tv_models.resnet152, 'IMAGENET1K_V1'),
        }

        model_fn, weights_name = resnet_map[model_type]
        if config.MODEL.PRETRAINED_TIMM:
            # Load with ImageNet pretrained weights
            model = model_fn(weights=weights_name)
            print(f"Loaded {model_type} with {weights_name} pretrained weights")
        else:
            model = model_fn(weights=None)

        # Replace the classifier head for the target number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
    elif model_type in ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4']:
        # EfficientNet models for teacher distillation
        import timm
        model = timm.create_model(
            model_type,
            pretrained=config.MODEL.PRETRAINED_TIMM,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
        )
    elif model_type.startswith('vit_'):
        # Vision Transformer models from timm
        import timm
        model = timm.create_model(
            model_type,
            pretrained=config.MODEL.PRETRAINED_TIMM,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            img_size=config.DATA.IMG_SIZE,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
