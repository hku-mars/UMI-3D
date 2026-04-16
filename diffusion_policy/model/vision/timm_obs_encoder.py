import copy
import math
import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x: NCHW
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        model_name: str,
        pretrained: bool,
        frozen: bool,
        global_pool: str,
        transforms: list,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization (assuming input in [0,1])
        imagenet_norm: bool = False,
        feature_aggregation: str = "spatial_embedding",
        downsample_ratio: int = 32,
        position_encording: str = "learnable",
        # NOTE: keep these args for Hydra compatibility; they are NO-OP in this lora-free version
        use_lora: bool = False,
        lora_rank: int = 8,
        drop_path_rate: float = 0.0,
        fused_model_name: str = "",
    ):
        """
        Assumes rgb input:     B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        if use_lora:
            logger.warning("use_lora=True is ignored in this lora-free timm_obs_encoder.py")
        if fused_model_name:
            logger.warning("fused_model_name is ignored in this simplified timm_obs_encoder.py")

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}
        key_eval_transform_map = nn.ModuleDict()

        # -----------------------
        # infer image_shape (H,W)
        # -----------------------
        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shp = tuple(attr["shape"])
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                # shp = (C,H,W)
                assert image_shape is None or image_shape == shp[1:], (
                    f"Inconsistent image shape in shape_meta: {image_shape} vs {shp[1:]}"
                )
                image_shape = shp[1:]
        assert image_shape is not None, "No rgb obs found in shape_meta['obs']"

        assert global_pool == ""

        # -----------------------
        # create timm model
        # -----------------------
        # Key fix for DINOv2 ViT: many timm dinov2 models default to img_size=518.
        # We force img_size to match dataset (e.g., 224) to avoid input-size assertion.
        create_kwargs = dict(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool,  # '' means no pooling
            num_classes=0,            # remove classification head
        )

        # Only pass img_size for models that support it (ViT/DeiT etc.).
        # It's safe for most transformer vision backbones in timm.
        if model_name.startswith("vit") or "dinov2" in model_name or "deit" in model_name:
            create_kwargs["img_size"] = image_shape[0]  # assume square or use H
            create_kwargs["drop_path_rate"] = drop_path_rate

        model = timm.create_model(**create_kwargs)

        if frozen:
            assert pretrained
            for p in model.parameters():
                p.requires_grad = False

        # -----------------------
        # feature dim / special handling
        # -----------------------
        feature_dim = None
        if model_name.startswith("resnet"):
            if downsample_ratio == 32:
                model = torch.nn.Sequential(*list(model.children())[:-2])
                feature_dim = 512
            elif downsample_ratio == 16:
                model = torch.nn.Sequential(*list(model.children())[:-3])
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith("convnext"):
            if downsample_ratio == 32:
                model = torch.nn.Sequential(*list(model.children())[:-2])
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith("vit") or "dinov2" in model_name or "deit" in model_name:
            # timm ViT returns tokens [B, N, C] when num_classes=0 and global_pool=''
            feature_dim = getattr(model, "num_features", None)
            if feature_dim is None:
                raise RuntimeError(f"Cannot infer num_features from model: {model_name}")

        if use_group_norm and (not pretrained):
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features,
                ),
            )

        # -----------------------
        # transforms
        # -----------------------
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            # hydra style: [{type: RandomCrop, ratio: ...}, ...]
            assert transforms[0].type == "RandomCrop"
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
            ] + transforms[1:]

        if imagenet_norm:
            # add imagenet norm at the end (assume input in [0,1])
            norm = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            norm = None

        train_tfms = [] if transforms is None else list(transforms)
        if norm is not None:
            train_tfms = train_tfms + [norm]
        transform = nn.Identity() if len(train_tfms) == 0 else nn.Sequential(*train_tfms)

        eval_tfms = [torchvision.transforms.Resize(size=image_shape[0], antialias=True)]
        if norm is not None:
            eval_tfms = eval_tfms + [norm]
        eval_transform = nn.Identity() if transforms is None else nn.Sequential(*eval_tfms)

        # -----------------------
        # assign per-key model/transform
        # -----------------------
        for key, attr in obs_shape_meta.items():
            shp = tuple(attr["shape"])
            typ = attr.get("type", "low_dim")
            key_shape_map[key] = shp
            if typ == "rgb":
                rgb_keys.append(key)
                key_model_map[key] = model if share_rgb_model else copy.deepcopy(model)
                key_transform_map[key] = transform
                key_eval_transform_map[key] = eval_transform
            elif typ == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {typ}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print("rgb keys:         ", rgb_keys)
        print("low_dim_keys keys:", low_dim_keys)

        # -----------------------
        # feature aggregation
        # -----------------------
        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_eval_transform_map = key_eval_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        # Keep original behavior: ViT uses CLS token, ignore other aggregations
        self.feature_aggregation = feature_aggregation
        if model_name.startswith("vit") or "dinov2" in model_name or "deit" in model_name:
            if self.feature_aggregation is not None:
                logger.warning(
                    f"ViT will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!"
                )
            self.feature_aggregation = None

        # For CNNs, keep original aggregations
        if not (model_name.startswith("vit") or "dinov2" in model_name or "deit" in model_name):
            # derive feature_map_shape only for CNN style features
            feature_map_shape = [x // downsample_ratio for x in image_shape]
            if self.feature_aggregation == "soft_attention":
                self.attention = nn.Sequential(
                    nn.Linear(feature_dim, 1, bias=False),
                    nn.Softmax(dim=1),
                )
            elif self.feature_aggregation == "spatial_embedding":
                self.spatial_embedding = nn.Parameter(
                    torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim)
                )
            elif self.feature_aggregation == "transformer":
                if position_encording == "learnable":
                    self.position_embedding = nn.Parameter(
                        torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim)
                    )
                elif position_encording == "sinusoidal":
                    num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                    pe = torch.zeros(num_features, feature_dim)
                    position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim)
                    )
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    self.register_buffer("position_embedding", pe, persistent=False)
                else:
                    raise NotImplementedError(f"Unsupported position_encording: {position_encording}")

                self.aggregation_transformer = nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                    num_layers=4,
                )
            elif self.feature_aggregation == "attention_pool_2d":
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_shape[0],
                    embed_dim=feature_dim,
                    num_heads=max(1, feature_dim // 64),
                    output_dim=feature_dim,
                )

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def aggregate_feature(self, feature):
        # ViT: feature is [B, N, C], use CLS token (N includes CLS as first token)
        if self.model_name.startswith("vit") or "dinov2" in self.model_name or "deit" in self.model_name:
            return feature[:, 0, :]

        # CNN: feature is [B, C, H, W]
        assert len(feature.shape) == 4

        if self.feature_aggregation == "attention_pool_2d":
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2)  # B, C, HW
        feature = torch.transpose(feature, 1, 2)       # B, HW, C

        if self.feature_aggregation == "avg":
            return torch.mean(feature, dim=1)
        elif self.feature_aggregation == "max":
            return torch.amax(feature, dim=1)
        elif self.feature_aggregation == "soft_attention":
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == "spatial_embedding":
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == "transformer":
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if isinstance(self.position_embedding, torch.Tensor) and self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos = torch.cat([zero_feature, feature], dim=1) + self.position_embedding
            feature_out = self.aggregation_transformer(feature_with_pos)
            return feature_out[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature

    def forward(self, obs_dict):
        features = []
        batch_size = next(iter(obs_dict.values())).shape[0]

        # rgb
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]

            img = img.reshape(B * T, *img.shape[2:])

            if self.training:
                img = self.key_transform_map[key](img)
            else:
                img = self.key_eval_transform_map[key](img)

            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)

            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # low-dim
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))

        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (1, attr["horizon"]) + shape,
                dtype=self.dtype,
                device=self.device,
            )
            example_obs_dict[key] = this_obs

        out = self.forward(example_obs_dict)
        assert len(out.shape) == 2 and out.shape[0] == 1
        return out.shape


if __name__ == "__main__":
    # simple smoke test
    pass
