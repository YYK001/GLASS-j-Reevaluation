"""Independent download/check/predict/evaluate/compare; no training entrypoint."""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from DINOv3.MADEqual.mvtec_broad6_compose2.runner import _atomic_json, _atomic_npz, _write_csv, _read_csv
from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import evaluate_fast, METRIC_NAMES
from DINOv3.MADEqual.mvtec_task_decoupled.runner import image_metrics
from DINOv3.MADEqual.normal_reconstruction.scoring import fixed_fpr_diagnostics
from external_baselines.glass_external.runner import summarize as shared_summary
from external_baselines.glass_external.official import SOURCE, ROOT
from . import checkpoints as cp
from .data import CATEGORIES, records, official_transform, load_rgb, crop_gt, reference_crop

SCOPE = 'official_fov_288_unified_metrics'


def categories(args):
    return tuple(CATEGORIES) if args.category == 'all' else (args.category,)


def device_for(name, *, metrics=False):
    device = torch.device(name)
    if metrics and device.type != 'cuda':
        raise ValueError('formal metrics require existing fast CUDA AUPRO; no CPU fallback')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


class Resources:
    def __init__(self, out, device):
        self.out, self.device, self.rows = out, device, []

    @contextmanager
    def measure(self, stage, category=''):
        cuda = self.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize(self.device)
            self.rows.append(dict(category=category, stage=stage, seconds=time.perf_counter()-start,
                gpu=torch.cuda.get_device_name(self.device) if cuda else 'CPU',
                allocated_peak_bytes=torch.cuda.max_memory_allocated(self.device) if cuda else 0,
                reserved_peak_bytes=torch.cuda.max_memory_reserved(self.device) if cuda else 0))
            _write_csv(self.out/'resources.csv', self.rows)


def state(out, args, **extra):
    return dict(protocol_id=cp.PROTOCOL, identity=cp.IDENTITY, stage=args.command, scope=SCOPE,
        categories=list(categories(args)), category_count=len(categories(args)),
        all_15_completed=set(categories(args)) == set(CATEGORIES), status='complete', **extra)


def summarize(metrics, fpr, selected):
    macro, fm = shared_summary(metrics, fpr, selected)
    for row in macro:
        row['method'] = metrics[0]['method']
    for row in fm:
        row['method'] = metrics[0]['method']
    return macro, fm


def download(args, out):
    cp.download(args.weights_root, categories(args))
    manifest = [cp.describe(cp.resolve(args.weights_root, c), c) for c in categories(args)]
    _write_csv(out/'checkpoint_manifest.csv', manifest)
    return state(out, args, strict_model_check='not_executed; run check next')


def check(args, out):
    from external_baselines.glass_external.adapter import model_for
    from external_baselines.glass_external.official import modules
    root, tests, counts = records(args.dataset_root, categories(args))
    paths = {c: cp.resolve(args.weights_root, c) for c in categories(args)}
    modules().utils.fix_seeds(0)
    model = model_for(args.backbone_weights, torch.device('cpu'))
    manifest, audit, geometry_rows = [], [], []
    for c in categories(args):
        loaded = cp.load_released(model, paths[c])
        manifest.append(cp.describe(paths[c], c, loaded))
        transform, resize = official_transform(tests[c])
        for record in tests[c]:
            _, geom = load_rgb(record, transform, resize)
            _, info = crop_gt(record, geom)
            key = dict(category=c, relative_path=Path(record.path).relative_to(root).as_posix())
            geometry_rows.append(dict(**key, **geom))
            audit.append(dict(**key, **info))
        print(f'[GLASS-RELEASE-CHECK] {c}: strict checkpoint load and test geometry checked', flush=True)
    _write_csv(out/'checkpoint_manifest.csv', manifest)
    _write_csv(out/'data_counts.csv', counts)
    _write_csv(out/'geometry.csv', geometry_rows)
    _write_csv(out/'mask_audit.csv', audit)
    _atomic_json(out/'backbone.json', dict(path=str(Path(args.backbone_weights).resolve()),
        sha256=cp.sha256(args.backbone_weights), source='torchvision Wide_ResNet50_2_Weights.IMAGENET1K_V1'))
    return state(out, args, strict_checkpoint_loading_passed=True, predictions_executed=False)


@torch.no_grad()
def infer(model, images):
    model.eval()
    scores, maps = model._predict(images.float())
    scores = np.asarray(scores).reshape(-1)
    maps = np.asarray(maps, dtype=np.float32)
    if scores.shape != (len(images),) or maps.shape != (len(images), 288, 288):
        raise ValueError('official prediction shape mismatch')
    if not np.isfinite(scores).all() or not np.isfinite(maps).all():
        raise FloatingPointError('nonfinite official prediction')
    return scores, maps


def predict(args, out):
    from external_baselines.glass_external.adapter import model_for
    from external_baselines.glass_external.official import modules
    root, tests, counts = records(args.dataset_root, categories(args))
    paths = {c: cp.resolve(args.weights_root, c) for c in categories(args)}
    device = device_for(args.device)
    meter = Resources(out, device)
    index, manifest, audits = [], [], []
    _write_csv(out/'data_counts.csv', counts)
    _atomic_json(out/'backbone.json', dict(path=str(Path(args.backbone_weights).resolve()),
        sha256=cp.sha256(args.backbone_weights), source='torchvision ImageNet V1'))
    for c in categories(args):
        modules().utils.fix_seeds(0)
        with meter.measure('model_loading', c):
            model = model_for(args.backbone_weights, device)
            loaded = cp.load_released(model, paths[c])
            manifest.append(cp.describe(paths[c], c, loaded))
            del loaded
        _write_csv(out/'checkpoint_manifest.csv', manifest)
        transform, resize = official_transform(tests[c])
        category_rows = []
        for start in range(0, len(tests[c]), args.batch_size):
            batch = tests[c][start:start+args.batch_size]
            with meter.measure('input_read_and_transform', c):
                inputs = [load_rgb(r, transform, resize) for r in batch]
                images = torch.stack([x for x, _ in inputs]).to(device)
            with meter.measure('official_model_inference', c):
                scores, maps = infer(model, images)
            with meter.measure('prediction_storage', c):
                for offset, (record, score, prediction, (_, geom)) in enumerate(zip(batch, scores, maps, inputs)):
                    rel = Path(record.path).relative_to(root).as_posix()
                    dest = f'{c}/maps/test_{start+offset:04d}.npz'
                    _atomic_npz(out/dest, relative_path=np.asarray(rel), image_score=np.asarray(score),
                                anomaly_map=prediction)
                    category_rows.append(dict(category=c, relative_path=rel, image_score=float(score),
                        prediction_file=dest, gt_file=f'{c}/gt/test_{start+offset:04d}.npz', **geom))
                # Partial prediction index remains available if a later batch fails.
                _write_csv(out/'sample_scores.csv', index+category_rows)
            del images, maps, scores, inputs
            print(f'[GLASS-RELEASE-PREDICT] {c}: {min(start+args.batch_size,len(tests[c]))}/{len(tests[c])}', flush=True)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # GT pixels are read only after all predictions of this category have been saved.
        with meter.measure('gt_prepare_and_storage', c):
            for record, row in zip(tests[c], category_rows):
                mask, audit = crop_gt(record, row)
                _atomic_npz(out/row['gt_file'], relative_path=np.asarray(row['relative_path']),
                            image_label=np.asarray(record.label, np.int64), mask=mask)
                audits.append(dict(category=c, relative_path=row['relative_path'], **audit))
                row['image_label'] = record.label
        index.extend(category_rows)
        _write_csv(out/'sample_scores.csv', index)
        _write_csv(out/'geometry.csv', [{k:v for k,v in r.items() if k not in
            ('image_score','prediction_file','gt_file','image_label')} for r in index])
        _write_csv(out/'mask_audit.csv', audits)
    return state(out, args, sample_count=len(index), strict_checkpoint_loading_passed=True,
                 predictions_fixed_before_gt_per_category=True, compare_executed=False)


def saved_inputs(root, selected):
    root = Path(root).resolve()
    meta = json.loads((root/'complete.json').read_text(encoding='utf-8'))
    if meta.get('protocol_id') != cp.PROTOCOL or meta.get('stage') != 'predict' or meta.get('status') != 'complete':
        raise ValueError('completed official-fov predict output required')
    if not set(selected) <= set(meta['categories']):
        raise ValueError('requested prediction categories are missing')
    rows = [r for r in _read_csv(root/'sample_scores.csv') if r['category'] in selected]
    keys = [(r['category'],r['relative_path']) for r in rows]
    if len(set(keys)) != len(rows) or set(r['category'] for r in rows) != set(selected):
        raise ValueError('duplicate/missing saved category or sample keys')
    counts = {r['category']:int(r['test_count']) for r in _read_csv(root/'data_counts.csv')}
    for c in selected:
        if sum(r['category'] == c for r in rows) != counts[c]:
            raise ValueError(f'missing saved samples: {c}')
    return root, rows


def load_sample(root, row):
    with np.load(root/row['prediction_file'], allow_pickle=False) as z:
        if str(z['relative_path']) != row['relative_path'] or float(z['image_score']) != float(row['image_score']):
            raise ValueError('saved prediction identity/score mismatch')
        prediction = z['anomaly_map'].copy()
    with np.load(root/row['gt_file'], allow_pickle=False) as z:
        if str(z['relative_path']) != row['relative_path'] or int(z['image_label']) != int(row['image_label']):
            raise ValueError('saved GT identity/label mismatch')
        mask = z['mask'].copy()
    if prediction.shape != (288,288) or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
        raise ValueError('saved prediction must be finite float32 288x288')
    if mask.shape != prediction.shape or not set(np.unique(mask)) <= {0,1}:
        raise ValueError('GT must be binary and aligned')
    return prediction, mask


def metric_rows(category, method, labels, scores, maps, masks, device, meter):
    with meter.measure(method+'_five_metric_evaluation', category):
        values = evaluate_fast(labels, masks, maps, device=device, allow_cpu_fallback=False)
        values.update(image_metrics(labels, scores))  # Official image scores, never map-max.
    with meter.measure(method+'_fixed_fpr_evaluation', category):
        fpr = [dict(category=category, method=method, **r) for r in fixed_fpr_diagnostics(maps, masks)]
    return dict(category=category, method=method, **values), fpr


def reference_index(args, sample_rows):
    """Explicit generic full-frame index or the existing frozen guided_validation output."""
    if args.reference_index:
        source = Path(args.reference_index).resolve()
        rows = _read_csv(source)
        for r in rows:
            if r.get('coordinate_system') != 'full_frame':
                raise ValueError('reference index must explicitly declare full_frame coordinates')
            path = Path(r['map_path'])
            r['map_path'] = str(path if path.is_absolute() else source.parent/path)
    else:
        root = Path(args.guided_reference).resolve()
        meta = json.loads((root/'complete.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'complete' or meta.get('protocol_id') != 'guided_validation_mvtec_seed42_v1':
            raise ValueError('completed MVTec guided_validation reference required')
        rows = [dict(r, map_path=str(root/r['category']/'maps'/f"test_{int(r['image_index']):04d}.npz"),
                     map_key=args.reference_method, image_score=r['FULL_LOCAL_MEAN'])
                for r in _read_csv(root/'sample_scores.csv')
                if r['split']=='test' and r['method']==args.reference_method]
    wanted = {(r['category'], r['relative_path']) for r in sample_rows}
    result = {}
    for r in rows:
        key = r['category'], r['relative_path']
        if key[0] not in {c for c,_ in wanted}:
            continue
        if key in result:
            raise ValueError(f'duplicate reference key: {key}')
        result[key] = r
    if set(result) != wanted:
        raise ValueError(f'reference identity mismatch; missing={sorted(wanted-set(result))[:10]}, extra={sorted(set(result)-wanted)[:10]}')
    return result


def evaluate(args, out):
    selected = categories(args)
    root, rows = saved_inputs(args.prediction_root, selected)
    device = device_for(args.device, metrics=True)
    meter = Resources(out, device)
    comparing = args.command == 'compare'
    refs = reference_index(args, rows) if comparing else None
    metrics, operating, transformed_rows = [], [], []
    for c in selected:
        group = [r for r in rows if r['category']==c]
        with meter.measure('saved_prediction_and_gt_read', c):
            pairs = [load_sample(root, r) for r in group]
            maps, masks = map(list, zip(*pairs))
            del pairs
            labels = [int(r['image_label']) for r in group]
            scores = [float(r['image_score']) for r in group]
        metric, fpr = metric_rows(c, cp.METHOD, labels, scores, maps, masks, device, meter)
        metrics.append(metric); operating.extend(fpr)
        del maps
        if comparing:
            maps, scores = [], []
            with meter.measure('reference_full_frame_read_crop_save', c):
                for i, row in enumerate(group):
                    r = refs[c, row['relative_path']]
                    path = Path(r['map_path'])
                    if path.suffix.lower()=='.npy':
                        array = np.load(path, allow_pickle=False)
                    else:
                        with np.load(path, allow_pickle=False) as z:
                            if 'relative_path' in z and str(z['relative_path']) != row['relative_path']:
                                raise ValueError('reference NPZ identity mismatch')
                            array = z[r['map_key']].copy()
                    geom = {k:int(v) for k,v in row.items() if k in
                        ('resized_height','resized_width','crop_top','crop_left')}
                    prediction = reference_crop(array, geom)
                    score = float(r['image_score'])
                    dest = f'{c}/reference_maps/test_{i:04d}.npz'
                    _atomic_npz(out/dest, anomaly_map=prediction, relative_path=np.asarray(row['relative_path']),
                                image_score=np.asarray(score))
                    transformed_rows.append(dict(category=c, relative_path=row['relative_path'],
                        image_score=score, prediction_file=dest, source_map=str(path), method=args.reference_method))
                    maps.append(prediction); scores.append(score)
            metric, fpr = metric_rows(c, 'REFERENCE_'+args.reference_method, labels, scores, maps, masks, device, meter)
            metrics.append(metric); operating.extend(fpr)
            del maps
            _write_csv(out/'reference_sample_scores.csv', transformed_rows)
        _write_csv(out/'category_metrics.csv', metrics)
        _write_csv(out/'fixed_fpr_diagnostics.csv', operating)
        del masks
        gc.collect(); torch.cuda.empty_cache()
        print(f'[GLASS-RELEASE-{args.command.upper()}] {c}: complete', flush=True)
    macro, fmacro = [], []
    for method in dict.fromkeys(r['method'] for r in metrics):
        m, f = summarize([r for r in metrics if r['method']==method],
                         [r for r in operating if r['method']==method], selected)
        macro.extend(m); fmacro.extend(f)
    _write_csv(out/'macro_metrics.csv', macro)
    _write_csv(out/'fixed_fpr_macro.csv', fmacro)
    if comparing:
        deltas = []
        for scope, table in [('category', metrics), ('macro', macro)]:
            for c in selected if scope=='category' else ('macro',):
                group = [r for r in table if scope=='macro' or r['category']==c]
                glass = next(r for r in group if r['method']==cp.METHOD)
                ref = next(r for r in group if r['method']!=cp.METHOD)
                deltas.append(dict(scope=scope, category=c, comparison='REFERENCE_'+args.reference_method+'_minus_'+cp.METHOD,
                    **{k:float(ref[k])-float(glass[k]) for k in METRIC_NAMES}))
        _write_csv(out/'common_fov_differences.csv', deltas)
    lines = ['# '+cp.IDENTITY, '', 'Scope: '+SCOPE, '',
             'AP=average precision. Macro=unweighted category mean; values in 0–1 units.', '',
             '| Method | Categories | '+ ' | '.join(METRIC_NAMES)+' |',
             '|---|---|'+ '|'.join(['---']*len(METRIC_NAMES))+'|']
    for r in macro:
        lines.append('| '+r['method']+' | '+str(r['category_count'])+' | '+
                     ' | '.join(f'{r[k]:.9f}' for k in METRIC_NAMES)+' |')
    lines += ['', 'Compare executed: '+str(comparing),
        'Only the declared categories were evaluated. No fixed640 claim; published model selection history is not independently established.',
        'GT is nearest-resized and center-cropped in the official FOV. No unobserved border scores were fabricated.',
        'Saved image scores retain each model’s original definition. Fixed-FPR thresholds are posthoc only.']
    (out/'objective_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return state(out, args, compare_executed=comparing, prediction_root=str(root), model_loaded=False)


def build_parser():
    parser = argparse.ArgumentParser(description=cp.IDENTITY)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('download','check','predict','evaluate','compare'):
        p = sub.add_parser(name)
        p.add_argument('--category', choices=('all',*CATEGORIES), default='all')
        p.add_argument('--output-dir', required=True)
        if name in ('download','check','predict'):
            p.add_argument('--weights-root', required=True)
        if name in ('check','predict'):
            p.add_argument('--dataset-root', required=True)
            p.add_argument('--backbone-weights', required=True)
        if name in ('predict','evaluate','compare'):
            p.add_argument('--device', default='cuda:0')
        if name == 'predict':
            p.add_argument('--batch-size', type=int, default=4)
        if name in ('evaluate','compare'):
            p.add_argument('--prediction-root', required=True)
        if name == 'compare':
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--reference-index', help='CSV: category,relative_path,map_path,map_key,image_score,coordinate_system=full_frame')
            group.add_argument('--guided-reference', help='completed existing MVTec guided_validation output')
            p.add_argument('--reference-method', default='GUIDED')
    return parser


class Tee:
    def __init__(self, stream, file): self.stream, self.file = stream, file
    def write(self, text): self.stream.write(text); self.file.write(text); self.file.flush()
    def flush(self): self.stream.flush(); self.file.flush()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args,'batch_size',1) < 1:
        raise ValueError('batch-size must be positive')
    out = Path(args.output_dir).resolve()
    from DINOv3.MADEqual.top9_vs_local.runner import _new_output
    paths = [Path(v).resolve() for k,v in vars(args).items() if v and k in
        ('weights_root','dataset_root','backbone_weights','prediction_root','reference_index','guided_reference')]
    _new_output(out, tuple(paths))
    with (out/'run.log').open('w',encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
        try:
            snapshot = out/'source_snapshot'
            snapshot.mkdir()
            for folder in (Path(__file__).parent, Path(__file__).parent.parent/'glass_external'):
                for path in folder.iterdir():
                    if path.suffix in ('.py','.json','.md','.txt'):
                        target = snapshot/folder.name/path.name
                        target.parent.mkdir(exist_ok=True); shutil.copy2(path,target)
            for path in list(ROOT.glob('*.py'))+[ROOT/'datasets/mvtec.py',ROOT/'LICENSE']:
                target = snapshot/'GLASS'/path.relative_to(ROOT)
                target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
            versions = {}
            for name in ('torch','torchvision','numpy','imgaug','timm','scipy','scikit-learn','gdown'):
                try: versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
            _atomic_json(out/'config.json', dict(protocol_id=cp.PROTOCOL, identity=cp.IDENTITY, scope=SCOPE,
                arguments=vars(args), command_line=sys.argv if argv is None else list(argv), source=SOURCE,
                publication=cp.RELEASE, versions=versions, precision='FP32', tf32=False, seed=0,
                model=dict(backbone='wideresnet50', backbone_weights='ImageNet V1', layers=['layer2','layer3'],
                    pretrain_embed_dimension=1536, target_embed_dimension=1536, patchsize=3,
                    dsc_layers=2, dsc_hidden=1024, pre_proj=1, imagesize=288,
                    gaussian_sigma=4, eval_mode=True, gradients=False),
                rgb='official MVTec TEST transform; toothbrush/wood resize329 else288; center crop288',
                gt='original binary mask -> nearest resize -> identical crop',
                checkpoint_selection='explicit published per-category file; no local training or selection',
                publication_selection_history='upstream best-selection uses test Image AUROC + Pixel AUROC; exact release history unknown'))
            result = (evaluate if args.command=='compare' else globals()[args.command])(args,out)
            _atomic_json(out/'complete.json', result)
        except Exception as exc:
            _atomic_json(out/'failure.json', dict(error=type(exc).__name__, message=str(exc)))
            (out/'exit_code').write_text('1\n'); traceback.print_exc(); raise
        else:
            (out/'exit_code').write_text('0\n')


if __name__ == '__main__':
    main()
