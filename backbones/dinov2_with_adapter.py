import torch
import torch.nn as nn
from transformers import AutoModel


class PiNoise(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=384):
        super().__init__()
        self.MLP1 = nn.Linear(in_dim, 2)
        self.MLP2 = nn.Linear(2, out_dim)
        nn.init.constant_(self.MLP1.weight, 0.)
        nn.init.constant_(self.MLP1.bias, 0.)
        nn.init.constant_(self.MLP2.weight, 0.)
        nn.init.constant_(self.MLP2.bias, 0.)

    def forward(self, x):
        return self.MLP2(self.MLP1(x)) + x


class Dinov2WithPiNoise(nn.Module):
    def __init__(self, model_path, hidden_dim=192, local_files_only=True):
        super().__init__()

        self.dino = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_files_only
        )

        self.out_dim = self.dino.config.hidden_size
        self.layer_num = self.dino.config.num_hidden_layers

        self.adapt_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # self.adapt_layers = [9, 10, 11]
        self.noise_maker = nn.ModuleDict({
            str(i): PiNoise(self.out_dim, self.out_dim, hidden_dim)
            for i in self.adapt_layers
        })

    def forward_features(self, x):
        hidden_states = self.dino.embeddings(pixel_values=x)

        for i, layer_module in enumerate(self.dino.encoder.layer):
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs
            if str(i) in self.noise_maker:
                hidden_states = self.noise_maker[str(i)](hidden_states)

        hidden_states = self.dino.layernorm(hidden_states)

        cls_feature = hidden_states[:, 0]
        return cls_feature

    def forward(self, x):
        return self.forward_features(x)