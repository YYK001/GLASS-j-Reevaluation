"""Defer heavyweight imports so --help works in the local Python-only workspace."""
import argparse
from pathlib import Path
from .protocol import EXCEL, selected


def parser():
    p = argparse.ArgumentParser(description='GLASS-j official VisA fixed-final reproduction')
    sub = p.add_subparsers(dest='stage', required=True)
    for stage in ('check','smoke','train','evaluate','summarize'):
        s = sub.add_parser(stage)
        s.add_argument('--output-dir', type=Path, required=True, help='NEW directory; existing outputs rejected')
        s.add_argument('--categories', nargs='+', default=['all'])
        if stage == 'summarize':
            s.add_argument('--result-roots', nargs='+', type=Path, required=True)
            continue
        s.add_argument('--device', default='cuda:0')
        s.add_argument('--workers', type=int, default=2)
        s.add_argument('--dataset-root', type=Path, required=stage != 'evaluate')
        s.add_argument('--excel', type=Path, default=EXCEL)
        s.add_argument('--backbone-weights', type=Path, required=stage != 'evaluate')
        if stage != 'evaluate':
            s.add_argument('--dtd-root', type=Path, required=True)
            s.add_argument('--foreground-root', type=Path, help='Directory containing <category>/<normal image filename>')
        if stage == 'smoke':
            s.set_defaults(categories=['candle'])
            s.add_argument('--smoke-images', type=int, default=8)
        if stage == 'evaluate':
            group = s.add_mutually_exclusive_group(required=True)
            group.add_argument('--trained-root', type=Path)
            group.add_argument('--prediction-root', type=Path, help='Offline re-evaluation from saved maps and GT')
            s.add_argument('--inference-batch', type=int, default=1)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.categories = selected(args.categories)
    if getattr(args, 'workers', 0) < 0:
        raise ValueError('workers must be non-negative')
    if args.stage == 'smoke' and (len(args.categories) != 1 or args.smoke_images < 2):
        raise ValueError('smoke requires one category and at least two normal images')
    from . import runner
    getattr(runner, args.stage)(args)
