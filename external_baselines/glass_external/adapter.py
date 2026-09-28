# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
from pathlib import Path
from unittest.mock import patch
import torch
from torchvision.models import _api
from .official import modules

MODEL_CONFIG = dict(layers_to_extract_from=['layer2','layer3'], input_shape=(3,288,288),
    pretrain_embed_dimension=1536,target_embed_dimension=1536,patchsize=3,meta_epochs=640,
    dsc_layers=2,dsc_hidden=1024,pre_proj=1,mining=1,noise=.015,radius=.75,p=.5,step=20,limit=392,
    train_backbone=False)


def model_for(weights, device):
    """Use official backbones.load; supply its ImageNet V1 state from an explicit local file."""
    weights = Path(weights).resolve()
    if not weights.is_file(): raise FileNotFoundError(f'local ImageNet wide_resnet50_2 V1 weights required: {weights}')
    official = modules()
    def local_state(*args,**kwargs):
        return torch.load(weights,map_location='cpu',weights_only=True)
    with patch.object(_api,'load_state_dict_from_url',local_state):
        backbone = official.backbones.load('wideresnet50')
    backbone.name,backbone.seed = 'wideresnet50',None
    model = official.glass.GLASS(device)
    model.load(backbone=backbone,device=device,**MODEL_CONFIG)
    model.backbone.requires_grad_(False)
    return model.to(device)
