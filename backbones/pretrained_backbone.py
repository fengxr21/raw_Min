from safetensors.torch import load_file
from torch import nn
import timm
import torch
from backbones.ViT_MiN import VisionTransformer
import  re
import math
from backbones.dinov2_with_adapter import Dinov2WithPiNoise

def convert_hf_dinov2_to_timm(hf_state_dict):
    """
    把 Hugging Face DINOv2 / ViT 风格的 state_dict
    转成 timm VisionTransformer 风格。
    """
    new_state_dict = {}

    qkv_cache = {}
    qkv_bias_cache = {}

    for k, v in hf_state_dict.items():
        new_k = None

        # -------------------------
        # embedding 部分
        # -------------------------
        if k == "embeddings.cls_token":
            new_k = "cls_token"
        elif k == "embeddings.position_embeddings":
            new_k = "pos_embed"
        elif k == "embeddings.patch_embeddings.projection.weight":
            new_k = "patch_embed.proj.weight"
        elif k == "embeddings.patch_embeddings.projection.bias":
            new_k = "patch_embed.proj.bias"

        # mask_token 在 timm 的这个模型里通常不需要
        elif k == "embeddings.mask_token":
            continue

        # 最后 LayerNorm
        elif k == "layernorm.weight":
            new_k = "norm.weight"
        elif k == "layernorm.bias":
            new_k = "norm.bias"

        # -------------------------
        # encoder block 部分
        # -------------------------
        elif k.startswith("encoder.layer."):
            # 例如 encoder.layer.3.norm1.weight
            m = re.match(r"encoder\.layer\.(\d+)\.(.*)", k)
            if m is None:
                continue

            layer_id = int(m.group(1))
            rest = m.group(2)

            prefix = f"blocks.{layer_id}"

            # norm
            if rest == "norm1.weight":
                new_k = f"{prefix}.norm1.weight"
            elif rest == "norm1.bias":
                new_k = f"{prefix}.norm1.bias"
            elif rest == "norm2.weight":
                new_k = f"{prefix}.norm2.weight"
            elif rest == "norm2.bias":
                new_k = f"{prefix}.norm2.bias"

            # MLP
            elif rest == "mlp.fc1.weight":
                new_k = f"{prefix}.mlp.fc1.weight"
            elif rest == "mlp.fc1.bias":
                new_k = f"{prefix}.mlp.fc1.bias"
            elif rest == "mlp.fc2.weight":
                new_k = f"{prefix}.mlp.fc2.weight"
            elif rest == "mlp.fc2.bias":
                new_k = f"{prefix}.mlp.fc2.bias"

            # attention output proj
            elif rest == "attention.output.dense.weight":
                new_k = f"{prefix}.attn.proj.weight"
            elif rest == "attention.output.dense.bias":
                new_k = f"{prefix}.attn.proj.bias"

            # layer scale
            # timm 某些 vit 版本叫 ls1.gamma / ls2.gamma
            elif rest == "layer_scale1.lambda1":
                new_k = f"{prefix}.ls1.gamma"
            elif rest == "layer_scale2.lambda1":
                new_k = f"{prefix}.ls2.gamma"

            # qkv 需要合并
            elif rest in [
                "attention.attention.query.weight",
                "attention.attention.key.weight",
                "attention.attention.value.weight",
            ]:
                qkv_cache.setdefault(layer_id, {})[rest.split(".")[-2]] = v
                continue

            elif rest in [
                "attention.attention.query.bias",
                "attention.attention.key.bias",
                "attention.attention.value.bias",
            ]:
                qkv_bias_cache.setdefault(layer_id, {})[rest.split(".")[-2]] = v
                continue

            else:
                # 其他 key 先跳过
                continue

        # 分类头忽略
        elif k.startswith("classifier") or k.startswith("head"):
            continue

        else:
            continue

        new_state_dict[new_k] = v

    # -------------------------
    # 合并 qkv
    # -------------------------
    for layer_id, parts in qkv_cache.items():
        if all(x in parts for x in ["query", "key", "value"]):
            new_state_dict[f"blocks.{layer_id}.attn.qkv.weight"] = torch.cat(
                [parts["query"], parts["key"], parts["value"]], dim=0
            )

    for layer_id, parts in qkv_bias_cache.items():
        if all(x in parts for x in ["query", "key", "value"]):
            new_state_dict[f"blocks.{layer_id}.attn.qkv.bias"] = torch.cat(
                [parts["query"], parts["key"], parts["value"]], dim=0
            )

    return new_state_dict

def get_pretrained_backbone(args):
    name = args['backbone_type']
    if name == "pretrained_vit_b16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        model.layer_num = 12
    elif name == "pretrained_vit_b16_224_in21k_min":
        model_f = timm.create_model("vit_base_patch16_224_in21k", pretrained=False, num_classes=0)

        safetensors_path = "/home/shengqin/ssd2ttt/fxr/Sup-21k VIT/model.safetensors"
        state_dict = load_file(safetensors_path)  # 返回 torch.Tensor dict
        missing, unexpected = model_f.load_state_dict(state_dict, strict=False)  # 加载到 timm 模型
        print("Missing keys after loading:", missing)
        print("Unexpected keys after loading:", unexpected)
        model = VisionTransformer(num_classes=0, weight_init='skip', args=args)
        model.load_state_dict(model_f.state_dict(), strict=False)
        model.out_dim = 768
        model.layer_num = 12
    elif name == "dinov2_min":
        from transformers import AutoModel
        print("Loading DINOv2 model: facebook/dinov2-base")
        model_path = '/home/shengqin/fxr/new_caprompt/few_shot/full_enhanced_prototype/dino-v2_enhanced_prototype_fscil_dataset/model/dinov2'
        model = Dinov2WithPiNoise(model_path, local_files_only=True)
    elif name == "pretrained_dinov2_b16_min":
        model_f = timm.create_model("vit_base_patch16_224.dino", pretrained=False ,checkpoint_path="/home/shengqin/ssd2ttt/fxr/MiN(few-shot+crs)/MiN/dino_vitbase16_pretrain.pth", num_classes=0)
        # safetensors_path = "/home/shengqin/fxr/new_caprompt/few_shot/full_enhanced_prototype/dino-v2_enhanced_prototype_fscil_dataset/model/dinov2/model.safetensors"
        # state_dict = load_file(safetensors_path)  # 返回 torch.Tensor dict
        # missing_keys, unexpected_keys = model_f.load_state_dict(state_dict, strict=False)
        # print("Missing keys:", missing_keys)
        # print("Unexpected keys:", unexpected_keys)
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            num_classes=0,
            weight_init='skip',
            args=args
        )
        model.load_state_dict(model_f.state_dict(), strict=False)
        model.out_dim = 768
        model.layer_num = 12
    elif name == "pretrained_dinov2_b16_min_beiyong":
        safetensors_path = "/home/shengqin/fxr/new_caprompt/few_shot/full_enhanced_prototype/dino-v2_enhanced_prototype_fscil_dataset/model/dinov2/model.safetensors"
        state_dict = load_file(safetensors_path)  # dict of torch.Tensor

        # 2. 创建你的自定义模型（保持原来的定义）
        model = VisionTransformer(
            img_size=224,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            num_classes=0,
            weight_init='skip',
            args=args
        )
        model.out_dim = 768
        model.layer_num = 12

        # 3. 构建键名映射表 (HuggingFace/DINOv2 -> 自定义模型)
        # 注意：自定义模型通常使用与 timm 类似的键名，但 patch_size=14 时 pos_embed 长度为 257
        key_mapping = {
            # 嵌入层
            'embeddings.cls_token': 'cls_token',
            'embeddings.position_embeddings': 'pos_embed',
            'embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
            'embeddings.patch_embeddings.projection.bias': 'patch_embed.proj.bias',
            # 最终 LayerNorm
            'layernorm.weight': 'norm.weight',
            'layernorm.bias': 'norm.bias',
        }

        # 为每个 Transformer 层添加映射（共12层）
        for i in range(12):
            # 注意力输出投影
            key_mapping[f'encoder.layer.{i}.attention.output.dense.weight'] = f'blocks.{i}.attn.proj.weight'
            key_mapping[f'encoder.layer.{i}.attention.output.dense.bias'] = f'blocks.{i}.attn.proj.bias'
            # LayerNorm
            key_mapping[f'encoder.layer.{i}.norm1.weight'] = f'blocks.{i}.norm1.weight'
            key_mapping[f'encoder.layer.{i}.norm1.bias'] = f'blocks.{i}.norm1.bias'
            key_mapping[f'encoder.layer.{i}.norm2.weight'] = f'blocks.{i}.norm2.weight'
            key_mapping[f'encoder.layer.{i}.norm2.bias'] = f'blocks.{i}.norm2.bias'
            # MLP
            key_mapping[f'encoder.layer.{i}.mlp.fc1.weight'] = f'blocks.{i}.mlp.fc1.weight'
            key_mapping[f'encoder.layer.{i}.mlp.fc1.bias'] = f'blocks.{i}.mlp.fc1.bias'
            key_mapping[f'encoder.layer.{i}.mlp.fc2.weight'] = f'blocks.{i}.mlp.fc2.weight'
            key_mapping[f'encoder.layer.{i}.mlp.fc2.bias'] = f'blocks.{i}.mlp.fc2.bias'

        # 4. 创建新的 state_dict，处理 QKV 合并
        new_state_dict = {}

        # 临时存储各层的 qkv 权重/偏置
        qkv_weights = {}  # {layer_idx: {'q': tensor, 'k': tensor, 'v': tensor}}
        qkv_biases = {}

        for src_key, tensor in state_dict.items():
            # 跳过不需要的键（mask_token, layer_scale 等）
            if 'mask_token' in src_key or 'layer_scale' in src_key:
                continue

            # 检查是否是 QKV 相关键
            if 'attention.attention.query.weight' in src_key:
                layer_idx = src_key.split('.')[2]  # 提取索引
                qkv_weights.setdefault(layer_idx, {})['q'] = tensor
            elif 'attention.attention.key.weight' in src_key:
                layer_idx = src_key.split('.')[2]
                qkv_weights.setdefault(layer_idx, {})['k'] = tensor
            elif 'attention.attention.value.weight' in src_key:
                layer_idx = src_key.split('.')[2]
                qkv_weights.setdefault(layer_idx, {})['v'] = tensor
            elif 'attention.attention.query.bias' in src_key:
                layer_idx = src_key.split('.')[2]
                qkv_biases.setdefault(layer_idx, {})['q'] = tensor
            elif 'attention.attention.key.bias' in src_key:
                layer_idx = src_key.split('.')[2]
                qkv_biases.setdefault(layer_idx, {})['k'] = tensor
            elif 'attention.attention.value.bias' in src_key:
                layer_idx = src_key.split('.')[2]
                qkv_biases.setdefault(layer_idx, {})['v'] = tensor
            else:
                # 普通键，直接映射
                if src_key in key_mapping:
                    dst_key = key_mapping[src_key]
                    new_state_dict[dst_key] = tensor
                else:
                    # 可选：打印未映射的键（调试用）
                    # print(f"Ignored key: {src_key}")
                    pass

        # 合并 QKV 权重
        for layer_idx, parts in qkv_weights.items():
            if 'q' in parts and 'k' in parts and 'v' in parts:
                q = parts['q']
                k = parts['k']
                v = parts['v']
                # 检查维度是否一致（假设都是 [D, embed_dim]）
                qkv_weight = torch.cat([q, k, v], dim=0)  # [3*D, embed_dim]
                new_state_dict[f'blocks.{layer_idx}.attn.qkv.weight'] = qkv_weight

        # 合并 QKV 偏置
        for layer_idx, parts in qkv_biases.items():
            if 'q' in parts and 'k' in parts and 'v' in parts:
                q = parts['q']
                k = parts['k']
                v = parts['v']
                qkv_bias = torch.cat([q, k, v], dim=0)  # [3*D]
                new_state_dict[f'blocks.{layer_idx}.attn.qkv.bias'] = qkv_bias

        # 5. 处理 pos_embed 形状适配（从 1370 变为 257）
        if 'pos_embed' in new_state_dict:
            src_pos = new_state_dict['pos_embed']  # shape: [1, 1370, 768]
            target_pos = model.pos_embed  # shape: [1, 257, 768]
            if src_pos.shape != target_pos.shape:
                print(f"Resizing pos_embed from {src_pos.shape} to {target_pos.shape}")

                # 计算原始和目标网格大小
                pretrain_patches = src_pos.shape[1] - 1  # 1369
                target_patches = target_pos.shape[1] - 1  # 256
                orig_hw = int(math.sqrt(pretrain_patches))  # 37
                target_hw = int(math.sqrt(target_patches))  # 16

                # 分离 cls token 和 patch embeddings
                cls_token = src_pos[:, :1, :]  # [1, 1, 768]
                patch_embeds = src_pos[:, 1:, :]  # [1, 1369, 768]

                # 将 patch_embeds 重塑为 [1, D, H, W]
                patch_embeds = patch_embeds.reshape(1, orig_hw, orig_hw, -1).permute(0, 3, 1, 2)  # [1, 768, 37, 37]

                # 双线性插值
                patch_embeds = torch.nn.functional.interpolate(
                    patch_embeds, size=(target_hw, target_hw), mode='bilinear', align_corners=False
                )  # [1, 768, 16, 16]

                # 恢复为 [1, target_patches, D]
                patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(1, target_patches, -1)  # [1, 256, 768]

                # 拼接 cls token
                new_state_dict['pos_embed'] = torch.cat([cls_token, patch_embeds], dim=1)  # [1, 257, 768]

        # 6. 加载权重到自定义模型
        # missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        # print("Missing keys after loading:", missing)
        # print("Unexpected keys after loading:", unexpected)
    else:
        raise 'No this type backbone!'

    return model


