"""Explicit published file identities, offline strict loading, optional official downloads."""
import hashlib
import json
from pathlib import Path
import re

import torch

RELEASE = json.loads(Path(__file__).with_name('release_index.json').read_text())
IDENTITY = 'GLASS-j official released checkpoints — MVTec AD re-evaluation'
METHOD = 'GLASS_J_RELEASED'
PROTOCOL = 'glass_j_mvtec_released_official_fov_288_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entry(category):
    entries = RELEASE['entries'][category]
    if len(entries) != 1 or not re.fullmatch(r'ckpt_best_\d+\.pth', entries[0]['filename']):
        raise ValueError(f'publication identity ambiguous: {category}')
    return entries[0]


def resolve(root, category):
    root = Path(root).resolve()
    filename = entry(category)['filename']
    # Only exact documented layouts; never take the first glob match.
    anchors = (root / 'results/models/backbone_0', root / 'models/backbone_0',
               root / 'backbone_0', root)
    found = {a / ('mvtec_' + category) / filename for a in anchors
             if (a / ('mvtec_' + category) / filename).is_file()}
    if len(found) != 1:
        raise FileNotFoundError(f'{category}: require exactly one published {filename}; found={sorted(map(str, found))}')
    path = found.pop()
    others = [p.name for p in path.parent.glob('*.pth') if p != path]
    if others:
        raise ValueError(f'{category}: additional checkpoint versions {others}; use a separate published-only directory')
    return path


def read_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f'checkpoint must be a state dictionary: {path}')
    return state


def strict_module(module, values):
    # PyTorch's BN backward-compatibility hook can silently supply a missing
    # num_batches_tracked even with strict=True. Require every actual key first.
    expected = module.state_dict()
    if not isinstance(values, dict) or values.keys() != expected.keys():
        raise ValueError('checkpoint module keys/buffers incomplete or unexpected')
    for key, value in values.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError(f'checkpoint tensor shape mismatch: {key}')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'nonfinite checkpoint tensor: {key}')
    module.load_state_dict(values, strict=True)


def load_released(model, path):
    """Keep official model_for construction order, then load both heads strictly like tester."""
    state = read_state(path)
    config = state.get('config', {})
    from external_baselines.glass_external.adapter import MODEL_CONFIG
    expected = dict(MODEL_CONFIG, imagesize=288, resize=288, seed=0, backbone='wideresnet50')
    if not isinstance(config, dict):
        raise ValueError('unrecognized embedded config')
    for key, value in config.items():
        if key in expected and json.dumps(value) != json.dumps(expected[key]):
            raise ValueError(f'checkpoint configuration differs: {key}={value}')
    if 'discriminator' not in state or 'pre_projection' not in state:
        raise ValueError('published checkpoint requires discriminator AND pre_projection; no partial loading')
    allowed = {'discriminator', 'pre_projection', 'backbone', 'backbone_buffers',
               'config', 'epoch', 'i_epoch', 'meta_epochs'}
    if set(state) - allowed:
        raise ValueError(f'unrecognized checkpoint entries: {set(state) - allowed}; inspect before loading')
    strict_module(model.discriminator, state['discriminator'])
    strict_module(model.pre_projection, state['pre_projection'])
    if 'backbone' in state:
        strict_module(model.backbone, state['backbone'])
    if 'backbone_buffers' in state:
        buffers = dict(model.backbone.named_buffers())
        if buffers.keys() != state['backbone_buffers'].keys():
            raise ValueError('backbone buffers incomplete')
        for k, value in buffers.items():
            source = state['backbone_buffers'][k]
            if value.shape != source.shape or value.dtype != source.dtype:
                raise ValueError(f'backbone buffer mismatch: {k}')
            value.copy_(source.to(value.device))
    model.float().eval().requires_grad_(False)
    return state


def describe(path, category, state=None):
    state = read_state(path) if state is None else state
    keys = {k: list(v)[:30] for k, v in state.items() if isinstance(v, dict)}
    return dict(category=category, path=str(Path(path).resolve()), size_bytes=Path(path).stat().st_size,
        sha256=sha256(path), top_level_keys=json.dumps(list(state)), state_dict_keys=json.dumps(keys),
        filename_epoch_index=int(re.fullmatch(r'ckpt_best_(\d+)\.pth', Path(path).name)[1]),
        embedded_epoch=state.get('epoch', state.get('i_epoch', 'unknown')),
        source=RELEASE['folder_url'], file_url='https://drive.google.com/file/d/'+entry(category)['file_id']+'/view',
        backbone_state='included' if 'backbone' in state else 'external ImageNet V1 + official construction probe',
        backbone_buffers='included' if 'backbone_buffers' in state or 'backbone' in state else 'not in checkpoint',
        selection_history='filename ckpt_best; upstream has test-AUROC best selection; exact publication history unavailable')


def download(root, categories):
    import gdown
    root = Path(root)
    if root.exists():
        raise FileExistsError(f'use a new independent download directory: {root}')
    root.mkdir(parents=True)
    for category in categories:
        item = entry(category)
        target = root / RELEASE['path_prefix'] / ('mvtec_' + category) / item['filename']
        target.parent.mkdir(parents=True)
        temporary = target.with_suffix('.pth.part')
        result = gdown.download(id=item['file_id'], output=str(temporary), quiet=False)
        if not result:
            raise RuntimeError(f'official download failed for {category}; use the README manual route')
        read_state(temporary)  # Detect HTML/error responses before publishing the file.
        temporary.rename(target)
