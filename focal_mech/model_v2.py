import sys
from typing import Optional
import torch

sys.path.append("../")
sys.path.append("../pretrain/")

from pretrain import AUDIO_CLIP
from dataset_v2 import MechDataset


class FCRelu(torch.nn.Module):
    def __init__(self,
                 in_features: int = 384,
                 out_features: int = 64,
                 module: Optional[torch.nn.Module] = None):
        super(FCRelu, self).__init__()
        if module is not None:
            self._module = module
        else:
            self._module = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.ReLU())

    def forward(self, x):
        return self._module(x)


class FocalMechanismClassifier(torch.nn.Module):
    def __init__(self,
                 pretrained_model: AUDIO_CLIP):
        super(FocalMechanismClassifier, self).__init__()
        self._pretrained_model = pretrained_model
        self._layers = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=0),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool1d(kernel_size=2, stride=2),

            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(12032, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
        )


    def forward(self, spec, station, mask):
        batch_size = spec.size(0)
        max_seq_len = spec.size(1)
        spec = spec.view(batch_size*max_seq_len, spec.size(2), spec.size(3), spec.size(4))
        spec = self._pretrained_model.encode_audio(spec)
        spec = spec.view(batch_size, max_seq_len, spec.size(1))

        x = torch.cat((spec, station), dim=-1)
        mask = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(~mask, 0.0)

        x = self._layers(x)
        x = torch.nn.Softmax(dim=-1)(x)
        return x


def build_model_from_pretrain(pretrained_model_path, device, mode="finetune") -> FocalMechanismClassifier:
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

    return FocalMechanismClassifier(pretrained_model)


def build_model(device: torch.device = torch.device("cpu")) -> FocalMechanismClassifier:
    pretrained_model = AUDIO_CLIP(
        embed_dim=384, text_input=8, text_width=512, text_layers=2,
        spec_tdim=600, spec_model_size='small224',
        device_name=device, imagenet_pretrain=False)
    return FocalMechanismClassifier(pretrained_model)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_paths = ["/data/jyzheng/SeisCLIP/datasets/mech/train.npy"]
    dataset = MechDataset(*train_paths)
    dataloader = DataLoader(dataset, batch_size=2)

    device = torch.device("cpu")
    pretrained_model_path = "/data/jyzheng/SeisCLIP/ckpt/pretrain/test/199.pt"
    model = build_model_from_pretrain(pretrained_model_path, device)

    for batch in dataloader:
        mask, spec, stat_info, label = batch
        output = model(spec, stat_info, mask)
        print(output.shape)
        print(output)
        break


