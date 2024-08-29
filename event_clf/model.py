import sys
from typing import Optional
import torch

sys.path.append("../")
sys.path.append("../pretrain/")

from pretrain import AUDIO_CLIP


class MLP(torch.nn.Module):
    def __init__(self, module: Optional[torch.nn.Module] = None):
        super(MLP, self).__init__()
        if module is not None:
            self._module = module
        else:
            self._module = torch.nn.Sequential(
                torch.nn.Linear(384, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 3),
                torch.nn.Softmax(dim=-1))

    def forward(self, x):
        return self._module(x)


class EventClassifier(torch.nn.Module):
    def __init__(self,
                 pretrained_model: AUDIO_CLIP,
                 mlp_model: torch.nn.Module):
        super(EventClassifier, self).__init__()
        self._pretrained_model = pretrained_model
        self._mlp_model = mlp_model

    def forward(self, spec):
        embed = self._pretrained_model.encode_audio(spec)
        output = self._mlp_model(embed)
        return output


def build_model_from_pretrain(pretrained_model_path, device, mode="freeze") -> EventClassifier:
    pretrained_model = AUDIO_CLIP(
        embed_dim=384, text_input=8, text_width=512, text_layers=2,
        spec_tdim=600, spec_model_size='small224',
        device_name=device, imagenet_pretrain=False)
    param = torch.load(pretrained_model_path, map_location=device)
    if mode == "freeze":
        pretrained_model.load_state_dict(param, strict=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False
    elif mode == "finetune":
        pretrained_model.load_state_dict(param, strict=True)
        for param in pretrained_model.parameters():
            param.requires_grad = True
    elif mode == "scratch":
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    mlp_model = MLP()
    return EventClassifier(pretrained_model, mlp_model)


def build_model(device: torch.device = torch.device("cpu")) -> EventClassifier:
    pretrained_model = AUDIO_CLIP(
        embed_dim=384, text_input=8, text_width=512, text_layers=2,
        spec_tdim=600, spec_model_size='small224',
        device_name=device, imagenet_pretrain=False)
    mlp_model = MLP()
    return EventClassifier(pretrained_model, mlp_model)
