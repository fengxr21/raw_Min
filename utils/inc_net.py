import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F
import scipy.stats as stats
import timm
import random


class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.feature_dim = self.backbone.out_dim
        self.fc = None

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    @staticmethod
    def generate_fc(in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {
            'features': hyper_features,
            'logits': logits
        }


class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        factory_kwargs = {"device": device, "dtype": torch.double}
        # self.W = torch.empty((self.out_features, self.in_features), **factory_kwargs)
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    # @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        # initiate params
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim  # dim of backbone
        self.class_prompt_initial_value = args['class_prompt_initial_value']
        self.prototype_pool = []
        self.enhanced_prototype_pool = []
        self.prompt_pool = []
        self.all_prompts = nn.Parameter(
            self.class_prompt_initial_value * torch.randn(args['num_classes'], self.feature_dim),
            requires_grad=True
        )
        if self.args['use_buffer']:
            self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        factory_kwargs = {"device": self.device, "dtype": torch.double}

        in_dim = self.buffer_size if self.args['use_buffer'] else self.feature_dim
        weight = torch.zeros((in_dim, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.Pinoise_list = nn.ModuleList()

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    def forward_fc(self, features):
        features = features.to(self.weight)
        return features @ self.weight

    @property
    def in_features(self) -> int:
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            if self.args['use_buffer']:
                fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
            else:
                fc = SimpleLinear(self.feature_dim, self.known_class, bias=False)
        else:
            if self.args['use_buffer']:
                fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            else:
                fc = SimpleLinear(self.feature_dim, nb_classes, bias=True)
        if self.normal_fc is None:
            self.normal_fc = fc
        else:
            nn.init.constant_(fc.weight, 0.)

            del self.normal_fc
            self.normal_fc = fc

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:

        X = self.buffer(self.backbone(X))

        X, Y = X.to(self.weight), Y.to(self.weight)

        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        self.R -= self.R @ X.T @ K @ X @ self.R
        self.weight += self.R @ X.T @ (Y - X @ self.weight)

    def forward(self, x, fc_only: bool = False,task_id=None,prompt_weight=None):

        hyper_features = self.backbone(x)
        if self.args['use_buffer']:
            logits = self.forward_fc(self.buffer(hyper_features))
        else:
            logits = self.forward_fc(hyper_features)
        return {
            'logits': logits
        }

    def save_raw_prototypes(self, task_id, class_ids, prototypes):
        if isinstance(class_ids, torch.Tensor):
            class_ids = class_ids.detach().cpu().tolist()

        item = {
            "task_id": int(task_id),
            "class_ids": list(class_ids),
            "prototypes": prototypes.detach().clone(),
        }
        self.prototype_pool.append(item)

    def get_current_task_raw_prototypes(self, task_id):
        task_id = int(task_id)

        for item in self.prototype_pool:
            if item["task_id"] == task_id:
                raw_prototypes = item["prototypes"].to(self.device)
                class_ids = item["class_ids"]
                return raw_prototypes, class_ids

        raise ValueError(f"Task {task_id} raw prototypes not found in prototype_pool.")

    def get_current_enhanced_prototypes(self, task_id):
        task_id = int(task_id)
        raw_prototypes, class_ids = self.get_current_task_raw_prototypes(task_id)
        enhanced_prototypes = raw_prototypes
        return enhanced_prototypes, class_ids

    def get_pre_enhanced_prototypes(self, task_id):
        task_id = int(task_id)

        proto_list = []
        class_id_list = []

        for item in self.enhanced_prototype_pool:
            if item["task_id"] < task_id:
                proto_list.append(item["enhanced_prototypes"].to(self.device))
                class_id_list.extend(item["class_ids"])

        if len(proto_list) == 0:
            return None, []

        all_pre_enhanced = torch.cat(proto_list, dim=0)
        return all_pre_enhanced, class_id_list

    def save_final_current_prompt(self, task_id, class_ids):
        if isinstance(class_ids, torch.Tensor):
            class_ids = class_ids.detach().cpu().tolist()

        class_ids_tensor = torch.tensor(
            class_ids, dtype=torch.long, device=self.all_prompts.device
        )
        learned_prompts = self.all_prompts[class_ids_tensor].detach().clone()

        item = {
            "task_id": int(task_id),
            "class_ids": list(class_ids),
            "prompts": learned_prompts
        }
        self.prompt_pool.append(item)

    def save_final_current_enhanced_prototypes(self, task_id, class_ids, enhanced_prototypes):
        if isinstance(class_ids, torch.Tensor):
            class_ids = class_ids.detach().cpu().tolist()

        item = {
            "task_id": int(task_id),
            "class_ids": list(class_ids),
            "enhanced_prototypes": enhanced_prototypes.detach().clone()
        }

        self.enhanced_prototype_pool.append(item)

    def get_all_previous_prototypes(self):
        if len(self.enhanced_prototype_pool) == 0:
            return None, []

        proto_list = []
        class_id_list = []

        for item in self.enhanced_prototype_pool:
            proto_list.append(item["enhanced_prototypes"].to(self.device))
            class_id_list.extend(item["class_ids"])

        all_prototypes = torch.cat(proto_list, dim=0)
        return all_prototypes, class_id_list

    def extract_feature(self, x):
        hyper_features = self.backbone(x)
        return hyper_features

    def forward_normal_fc(self, x, use_prompt: bool = False,task_id=None,prompt_weight=None):

        # if use_prompt:
        hyper_features = self.backbone(x)
        # else:
        #     hyper_features = self.backbone(x)
        # return hyper_features
        if self.args['use_buffer']:
            hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(torch.float32)
        logits = self.normal_fc(hyper_features)['logits']
        return {
            "logits": logits
        }

    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[j].parameters():
                param.requires_grad = True
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        #
        # for i in self.backbone.adapt_layers:
        #     for param in self.backbone.noise_maker[str(i)].parameters():
        #         param.requires_grad = True
        #
        #     for p in self.backbone.dino.encoder.layer[i].norm1.parameters():
        #         p.requires_grad = True
        #
        #     for p in self.backbone.dino.encoder.layer[i].norm2.parameters():
        #         p.requires_grad = True
        #
        # for p in self.backbone.dino.layernorm.parameters():
        #     p.requires_grad = True
