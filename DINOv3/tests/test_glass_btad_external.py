# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
import torch
from external_baselines.glass_external import adapter
from external_baselines.glass_external.official import modules

class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=torch.nn.Conv2d(3,8,3,stride=4,padding=1)
        self.layer2=torch.nn.Sequential(torch.nn.Conv2d(8,8,3,stride=2,padding=1),torch.nn.BatchNorm2d(8))
        self.layer3=torch.nn.Sequential(torch.nn.Conv2d(8,12,3,stride=2,padding=1),torch.nn.BatchNorm2d(12))
    def forward(self,x):return self.layer3(self.layer2(self.stem(x)))


def small_model(device='cpu',real_backbone=False):
    modules().utils.fix_seeds(0)
    if real_backbone:
        from torchvision.models import wide_resnet50_2
        backbone=wide_resnet50_2(weights=None)
    else: backbone=TinyBackbone()
    model=modules().glass.GLASS(torch.device(device))
    cfg=dict(adapter.MODEL_CONFIG,input_shape=(3,64,64),meta_epochs=1,limit=0)
    if not real_backbone: cfg.update(pretrain_embed_dimension=16,target_embed_dimension=16,dsc_hidden=16)
    model.load(backbone=backbone,device=torch.device(device),**cfg)
    model.backbone.requires_grad_(False)
    return model
