import torch.nn as nn


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

        elif isinstance(m, nn.Parameter):
            pass  # since only lora use nn.Parameter and it has its own initialization

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            nn.init.zeros_(m.bias.data)
