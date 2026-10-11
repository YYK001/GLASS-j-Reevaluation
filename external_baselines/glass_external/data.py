# Exported shared helpers; bodies unchanged. See CODE_PROVENANCE.json.
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

def geometry(path, size=288):
    with Image.open(path) as image:
        w,h = image.size
        # Ask the same torchvision Resize operation for exact integer rounding.
        rw,rh = TF.resize(Image.new('RGB',(w,h)),size).size
    top,left = int(round((rh-size)/2.)),int(round((rw-size)/2.))
    return dict(original_height=h, original_width=w, resized_height=rh,resized_width=rw,
        crop_top=top,crop_left=left,crop_size=size,output_height=h//4,output_width=w//4,
        outside_crop='replicate nearest crop boundary; unobserved image area')


def map_to_evaluation(prediction, geom):
    value = np.asarray(prediction,np.float32)
    size = geom['crop_size']
    if value.shape != (size,size) or not np.isfinite(value).all():
        raise ValueError('official continuous map shape/numerics invalid')
    top,left = geom['crop_top'],geom['crop_left']
    rh,rw = geom['resized_height'],geom['resized_width']
    x = torch.from_numpy(value)[None,None]
    # Restore crop location first. Direct stretching would assign the central crop to the whole image.
    x = F.pad(x,(left,rw-left-size,top,rh-top-size),mode='replicate')
    return F.interpolate(x,size=(geom['output_height'],geom['output_width']),
        mode='bilinear',align_corners=False)[0,0].numpy()
