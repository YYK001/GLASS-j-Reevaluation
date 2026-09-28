"""Load upstream's flat modules without leaking their names into other baselines."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]/'GLASS'
SOURCE = json.loads(Path(__file__).with_name('source.json').read_text(encoding='utf-8'))


@lru_cache(None)
def modules():
    names = ('loss','model','common','metrics','utils','perlin','backbones','glass','mvtec')
    previous = {n:sys.modules.get(n) for n in names}
    loaded = {}
    try:
        for name in names:
            path = ROOT/('datasets/mvtec.py' if name == 'mvtec' else name+'.py')
            unique = __package__+'._official_'+name
            spec = importlib.util.spec_from_file_location(unique,path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, value in previous.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name] = value
    return SimpleNamespace(**loaded)
