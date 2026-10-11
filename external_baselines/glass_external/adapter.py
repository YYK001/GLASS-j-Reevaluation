# Exported shared helpers; bodies unchanged. See CODE_PROVENANCE.json.
from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torchvision.models import _api
from .official import modules,SOURCE

PROTOCOL = 'glass_j_btad_final640_seed0_v1'


MODEL_CONFIG = dict(layers_to_extract_from=['layer2','layer3'], input_shape=(3,288,288),
    pretrain_embed_dimension=1536,target_embed_dimension=1536,patchsize=3,meta_epochs=640,
    dsc_layers=2,dsc_hidden=1024,pre_proj=1,mining=1,noise=.015,radius=.75,p=.5,step=20,limit=392,
    train_backbone=False)


@contextmanager
def working_directory(path):
    before = Path.cwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(before)


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


class ScalarLog:
    """Forward official tensorboard writes and collect scalar losses without changing learning."""
    def __init__(self, wrapped):
        self.wrapped=wrapped; self.rows=[]
    def add_scalar(self,tag,value,step,*args,**kwargs):
        v=float(value)
        if not np.isfinite(v): raise FloatingPointError(f'nonfinite official training scalar: {tag}')
        self.rows.append(dict(step=int(step),tag=tag,value=v))
        return self.wrapped.add_scalar(tag,value,step,*args,**kwargs)
    def close(self): self.wrapped.close()


def train_last(model, loader, category, output, *, formal, protocol_id=PROTOCOL, dataset_prefix='btad'):
    if formal and (model.meta_epochs != 640 or model.limit != 392):
        raise ValueError('formal training requires 640 official meta epochs and limit=392')
    if loader.dataset.distribution not in (2,3): raise ValueError('normal distribution decision required first')
    # This is the unmodified official epoch loop and _train_discriminator, without val_data.
    with working_directory(output):
        modules().glass.GLASS.trainer(model,loader,None,dataset_prefix+'_'+category,final_only=True)
    state = torch.load(Path(model.ckpt_dir)/'ckpt.pth',map_location='cpu',weights_only=True)
    state.update(protocol_id=protocol_id, source_commit=SOURCE['commit'], category=category,
        completed_epochs=model.meta_epochs,formal_model=formal,distribution=int(model.svd),
        # Preserve BN buffers from the official feature-dimension probe as well as frozen weights.
        backbone_buffers={k:v.detach().cpu() for k,v in model.backbone.named_buffers()})
    path=Path(output)/'ckpt_last.pth'
    torch.save(state,path)
    return path


def load_last(model, path, category, *, require_formal=True, protocol_id=PROTOCOL):
    state=torch.load(path,map_location=model.device,weights_only=True)
    if (state.get('protocol_id')!=protocol_id or state.get('source_commit')!=SOURCE['commit'] or
        state.get('category')!=category): raise ValueError('checkpoint protocol/source/category mismatch')
    if require_formal and (not state.get('formal_model') or state.get('completed_epochs')!=640):
        raise ValueError('evaluation requires the fixed 640th epoch, not smoke or best checkpoint')
    model.discriminator.load_state_dict(state['discriminator'])
    model.pre_projection.load_state_dict(state['pre_projection'])
    buffers=dict(model.backbone.named_buffers())
    if buffers.keys()!=state['backbone_buffers'].keys(): raise ValueError('backbone buffer mismatch')
    for k,v in buffers.items(): v.copy_(state['backbone_buffers'][k].to(v.device))
    model.svd=int(state['distribution'])
    return state
