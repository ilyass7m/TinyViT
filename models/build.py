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
        # Legacy CLIP model (no pretrained weights)
        from .clip import CLIP
        kwargs = {
            'embed_dim': 768, 'image_resolution': 224,
            'vision_layers': 24, 'vision_width': 1024, 'vision_patch_size': 14,
            "num_classes": config.MODEL.NUM_CLASSES,
        }
        model = CLIP(**kwargs)
    elif model_type == 'clip_vit_large_patch14':
        # CLIP-ViT-L/14 with pretrained weights from OpenCLIP
        # This is the model used in the TinyViT paper for distillation
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for pretrained CLIP models. "
                "Install with: pip install open_clip_torch"
            )
        import torch.nn as nn

        # Determine which pretrained weights to use
        # PRETRAINED_TIMM=True means load pretrained weights
        # Default to 'openai' which is the original CLIP weights
        if config.MODEL.PRETRAINED_TIMM:
            pretrained = 'openai'  # Default to OpenAI weights
        else:
            pretrained = None

        print(f"Loading CLIP-ViT-L/14 with pretrained={pretrained}")

        # Load CLIP model from OpenCLIP
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained=pretrained,
        )

        # Create a wrapper that adds classification head
        class CLIPVisionClassifier(nn.Module):
            """CLIP visual encoder with classification head."""
            def __init__(self, clip_model, num_classes, freeze_visual=False):
                super().__init__()
                self.visual = clip_model.visual
                # ViT-L/14 has output dim of 768
                self.head = nn.Linear(768, num_classes)
                nn.init.zeros_(self.head.bias)
                nn.init.trunc_normal_(self.head.weight, std=0.02)

                # Optionally freeze visual encoder (for head-only finetuning)
                self.freeze_visual = freeze_visual
                if freeze_visual:
                    for param in self.visual.parameters():
                        param.requires_grad = False
                    # Set visual encoder to eval mode permanently
                    self.visual.eval()
                    print("CLIP visual encoder frozen - only training classification head")

            def train(self, mode=True):
                """Override train to keep visual encoder in eval mode when frozen."""
                super().train(mode)
                if self.freeze_visual:
                    # Always keep visual encoder in eval mode
                    self.visual.eval()
                return self

            def forward(self, x):
                features = self.visual(x)
                return self.head(features)

            def forward_features(self, x):
                return self.visual(x)

        # Freeze visual encoder when finetuning (EVAL_BN_WHEN_TRAINING as proxy flag)
        freeze_visual = getattr(config.TRAIN, 'EVAL_BN_WHEN_TRAINING', False)
        model = CLIPVisionClassifier(clip_model, config.MODEL.NUM_CLASSES, freeze_visual=freeze_visual)
        print(f"Built CLIP-ViT-L/14 classifier with {config.MODEL.NUM_CLASSES} classes")
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

        # Only replace the classifier head if num_classes differs from pretrained (1000)
        if config.MODEL.NUM_CLASSES != 1000:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.MODEL.NUM_CLASSES)
            print(f"Replaced fc layer for {config.MODEL.NUM_CLASSES} classes")
        else:
            print(f"Keeping pretrained fc layer for {config.MODEL.NUM_CLASSES} classes")
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
