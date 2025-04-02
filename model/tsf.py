from functools import partial
import torch.nn as nn
from .vit import VisionTransformer


def tsf_base_4c_32f(num_classes):
    model = VisionTransformer(
        img_size=224,
        num_classes=num_classes,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        num_frames=32,
        attention_type='divided_space_time',
    )
    return model
