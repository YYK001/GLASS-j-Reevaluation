"""Thin VisA wrapper: upstream learns/predicts; existing project functions evaluate."""
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
from contextlib import contextmanager

from external_baselines.patchcore_official_eval.storage import json_write, csv_write, npz_write, read_json
from .protocol import (CATEGORIES, COUNTS, EXCEL, METHOD, PROTOCOL, SOURCE, FOREGROUND_URL,
                       WEIGHT_URL, WEIGHT_SHA256, branches, foreground_inventory)


def new_output(args):
    out = args.output_dir.expanduser().resolve()
    # Do not put outputs into, above, or on top of an input or the checked-in source.
    inputs = [Path(__file__).resolve().parents[2]]
    inputs += [Path(v).resolve() for k,v in vars(args).items()
               if k.endswith(('root','weights','excel')) and v is not None]
    inputs += [Path(p).resolve() for p in getattr(args, 'result_roots', [])]
    for p in inputs:
        if out == p or out in p.parents or p in out.parents:
            raise ValueError('Output would contain/replace an input: '+str(p))
    if out == inputs[0] or inputs[0] in out.parents:
        raise ValueError('Keep experiment outputs outside external_baselines')
    out.mkdir(parents=True, exist_ok=False)
    json_write(out/'state.json', dict(status='running', stage=args.stage, protocol_id=PROTOCOL,
                                    categories=list(args.categories)))
    return out


@contextmanager
def operation(args):
    out = new_output(args)
    try:
        yield out
    except Exception as e:
        json_write(out/'failure.json', dict(status='failed', type=type(e).__name__, message=str(e),
            protocol_id=PROTOCOL, stage=args.stage))
        raise


class Tee:
    def __init__(self, console, file): self.console, self.file = console, file
    def write(self, value):
        self.console.write(value); self.file.write(value); self.file.flush()
        return len(value)
    def flush(self): self.console.flush(); self.file.flush()
    def isatty(self): return False


@contextmanager
def logging(path):
    from contextlib import redirect_stdout, redirect_stderr
    with Path(path).open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
            yield


def runtime():
    names = ('torch','torchvision','numpy','imgaug','timm','tensorboard','pandas','openpyxl',
             'Pillow','scipy','scikit-learn','opencv-python-headless','scikit-image','tqdm')
    versions, missing = {}, []
    for n in names:
        try: versions[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: missing.append(n)
    return dict(python=sys.version, executable=sys.executable, versions=versions, missing=missing)


def require_runtime(out):
    env = runtime(); json_write(out/'environment.json', env)
    if env['missing']:
        raise RuntimeError('Use the independent GLASS environment; missing: '+', '.join(env['missing']))
    import torch
    from external_baselines.glass_external.official import modules
    modules()  # Import compatibility, not just package metadata.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def device_for(name):
    from external_baselines.glass_external.runner import cuda
    return cuda(name)


def weight_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError('Expected official ImageNet V1 wide_resnet50_2-95faca4d.pth; weight identity mismatch')
    return dict(path=str(path), source=WEIGHT_URL, sha256=digest.hexdigest(), kind='ImageNet V1')


def configuration(args, table, weight=None):
    from external_baselines.glass_external.adapter import MODEL_CONFIG
    from external_baselines.glass_external.official import modules
    defaults = {k:v.default for k,v in inspect.signature(modules().glass.GLASS.load).parameters.items()
                if v.default is not inspect.Parameter.empty}
    defaults.update(MODEL_CONFIG)
    dataset = {k:v.default for k,v in inspect.signature(modules().visa.VisADataset.__init__).parameters.items()
               if v.default is not inspect.Parameter.empty and k not in ('split','kwargs')}
    dataset.update(source='', anomaly_source_path=str(args.dtd_root), classname='see per-category config',
                   dataset_name='visa', seed=0, batch_size=8, resize=288, imagesize=288,
                   fg=1, distribution=0, mean=.5, std=.1, rand_aug=1)
    return dict(protocol_id=PROTOCOL, method=METHOD, source=SOURCE,
        arguments={k:([str(x) for x in v] if isinstance(v,(tuple,list)) else str(v) if isinstance(v,Path) else v)
                   for k,v in vars(args).items()}, categories=list(args.categories), model=defaults, dataset=dataset,
        backbone_weight=weight, branches={c:table[c] for c in args.categories},
        training='all official 1cls.csv normal train images; no calibration exclusion; DTD auxiliary only',
        checkpoint='fixed meta epoch 640; final_only=True; no validation or best checkpoint selection',
        limit_semantics='official center uses entire loader; discriminator stops AFTER sample_num > 392 update',
        inference='official GLASS._predict: original patch image score, native bilinear map and scipy Gaussian sigma=4',
        geometry='short-side Resize(288), CenterCrop(288); replicate crop boundary onto resized canvas; bilinear align_corners=False to original H//4,W//4',
        gt='original CSV-associated nonzero defect IDs; existing nearest-neighbor full-image H//4,W//4 mask adapter',
        evaluation='project fast CUDA AUPRO 200 thresholds FPR 0.3, four-connected; average precision; category-equal macro',
        fixed_fpr='caps 0.01 and 0.05; four-connected; small region area <= 0.001 * evaluation image area; N/A excluded with valid category denominator',
        differences=['fixed last epoch instead of original best test-selected checkpoint',
                     'project full-image H//4,W//4 evaluation with crop-exterior boundary extension',
                     'project fast CUDA metrics; not official metric bitwise reproduction',
                     'absolute validated Excel lookup mapped to equivalent explicit official modes 2/3',
                     'normal foreground paths resolved by explicit path hook; synthesis unchanged'])


def inputs(args, out):
    from PIL import Image
    import numpy as np
    from .data import records, metadata
    root, train, test, counts = records('visa', args.dataset_root, args.categories)
    csv_write(out/'data_counts.csv', [dict(r, expected=list(COUNTS[r['category']]),
        matches_reference=(r['training_normal'],r['test_normal'],r['test_anomaly']) == COUNTS[r['category']])
        for r in counts])
    manifest = [metadata(r,root,role) for c in args.categories
                for role,rs in (('normal_training',train[c]),('test',test[c])) for r in rs]
    # metadata()'s PatchCore input size is replaced with GLASS's true size and geometry.
    from .data import geometry
    for row in manifest:
        row.update(input_hw=[288,288], geometry=geometry(root/row['relative_path']),
                   coordinate_rule='GLASS center crop restored with boundary extension; original H//4,W//4')
    json_write(out/'data_manifest.json', manifest); csv_write(out/'data_manifest.csv', manifest)
    mismatches = [r for r in counts if (r['training_normal'],r['test_normal'],r['test_anomaly']) != COUNTS[r['category']]]
    if mismatches:
        raise ValueError('VisA counts differ from official reference; inspect data_counts.csv and CSV paths. No images removed: '+str(mismatches))
    table = branches(args.excel)
    csv_write(out/'branches.csv', [table[c] for c in args.categories])
    fgroot = Path(args.foreground_root or root/'fg_mask').resolve()
    mapping, fgrows, issues = foreground_inventory([r for c in args.categories for r in train[c]], fgroot)
    csv_write(out/'foreground_manifest.csv', fgrows)
    for row in fgrows:
        if row['status'] != 'matched': continue
        with Image.open(row['image_path']) as im, Image.open(row['foreground_path']) as mask:
            row.update(image_hw=[im.height,im.width], foreground_hw=[mask.height,mask.width], foreground_mode=mask.mode)
            values = np.asarray(mask)
            # The official path consumes channel 0 without changing encoding. Require an
            # aligned 8-bit mask; ceil(ToTensor) retains the original foreground semantics.
            if im.size != mask.size or mask.mode not in ('1','L','RGB') or values.dtype not in (np.uint8,np.bool_):
                issues.append(dict(row, status='invalid_geometry_or_encoding'))
            else:
                from torchvision import transforms
                # Match official foreground transforms exactly. An empty cropped foreground
                # would make upstream perlin_mask retry forever; fail here instead.
                transform = transforms.Compose([transforms.Resize(288),transforms.CenterCrop(288),transforms.ToTensor()])
                if not bool(transform(mask)[0].gt(0).any()):
                    issues.append(dict(row, status='empty_foreground_after_official_crop'))
    csv_write(out/'foreground_manifest.csv', fgrows)
    json_write(out/'foreground_issues.json', dict(root=str(fgroot), issues=issues,
        official_preparation=FOREGROUND_URL, fg=1, fallback_to_fg0=False))
    if issues:
        raise FileNotFoundError(f'{len(issues)} missing/ambiguous/invalid normal foreground masks; see foreground_issues.json. Official release: {FOREGROUND_URL}')
    dtd = Path(args.dtd_root).resolve()
    textures = sorted(dtd.glob('*/*.jpg'))
    if not textures: raise FileNotFoundError('DTD images/<texture>/*.jpg not found: '+str(dtd))
    for path in textures:
        with Image.open(path) as image: image.verify()
    weight = weight_identity(args.backbone_weights)
    cfg = configuration(args, table, weight)
    cfg.update(dataset_root=str(root), dtd_root=str(dtd), dtd_texture_count=len(textures), foreground_root=str(fgroot))
    json_write(out/'config.json', cfg)
    return root, train, test, table, mapping, cfg


def complete(out, args, **extra):
    json_write(out/'complete.json', dict(status='complete', protocol_id=PROTOCOL, method=METHOD,
        stage=args.stage, categories=list(args.categories), **extra))


def check(args):
    with operation(args) as out:
        torch = require_runtime(out)
        root, train, test, table, mapping, cfg = inputs(args,out)
        from .data import VisADataset
        from torchvision.models import wide_resnet50_2
        # Strict backbone validation is CPU-only and performs no learning/download.
        model = wide_resnet50_2(weights=None)
        model.load_state_dict(torch.load(args.backbone_weights,map_location='cpu',weights_only=True),strict=True)
        del model
        for c in args.categories:
            ds = VisADataset(train[c],args.dtd_root,training=True,foreground=mapping,
                             distribution_mode=table[c]['explicit_distribution_mode'])
            assert ds.class_fg == 1 and len(ds) == len(train[c])
            print(c, dict(train_normal=len(train[c]), test_normal=sum(r.label==0 for r in test[c]),
                          test_anomaly=sum(r.label==1 for r in test[c])), table[c], flush=True)
        complete(out,args,model_initialized='CPU backbone for strict weight check only', cuda_available=torch.cuda.is_available(),
                 actual_cuda_training_or_inference='not_executed')


def train(args):
    _train(args, smoke=False)


def smoke(args):
    _train(args, smoke=True)


def _train(args, smoke):
    with operation(args) as out:
        torch = require_runtime(out); device = device_for(args.device)
        root, train, test, table, mapping, cfg = inputs(args,out)
        from .data import VisADataset
        from external_baselines.glass_external.adapter import model_for, train_last, load_last, ScalarLog
        from external_baselines.glass_external.runner import loader
        from external_baselines.glass_mvtec_pretrained.runner import Resources
        from external_baselines.glass_external.official import modules
        import numpy as np
        for c in args.categories:
            target = out/c; target.mkdir()
            meter = Resources(target,device)
            with logging(target/'train.log'):
                modules().utils.fix_seeds(0)  # Every category starts from the same official seed.
                items = train[c][:args.smoke_images] if smoke else train[c]
                ds = VisADataset(items,args.dtd_root,training=True,foreground=mapping,
                                 distribution_mode=table[c]['explicit_distribution_mode'])
                dl = loader(ds,args.workers,training=True)
                json_write(target/'config.json',dict(cfg, category=c, smoke=smoke,
                    dataset=dict(cfg['dataset'],classname=c,distribution=ds.distribution,
                                 requested_official_distribution=0,split='train'),
                    model=dict(cfg['model'],meta_epochs=1 if smoke else 640,limit=0 if smoke else 392),
                    actual_meta_epochs=1 if smoke else 640, actual_limit=0 if smoke else 392,
                    actual_training_count=len(items)))
                with meter.measure('official_model_initialization',c):
                    model = model_for(args.backbone_weights,device)
                    if smoke: model.meta_epochs,model.limit = 1,0
                    model.set_model_dir(str(target/'official'), 'visa_'+c)
                before = {k:v.detach().clone() for k,v in model.pre_projection.named_parameters()} if smoke else None
                scalar = ScalarLog(model.logger.logger); model.logger.logger = scalar
                try:
                    with meter.measure('official_training_and_final_save',c):
                        checkpoint = train_last(model,dl,c,target,formal=not smoke,
                                                protocol_id=PROTOCOL,dataset_prefix='visa')
                finally:
                    csv_write(target/'training_scalars.csv',scalar.rows); scalar.close()
                    csv_write(target/'foreground_reads_main_process.csv',ds.foreground_reads)
                if int(model.svd) != table[c]['excel_distribution']:
                    raise RuntimeError('Trainer did not apply the official Excel branch')
                updates = len([r for r in scalar.rows if r['tag']=='loss'])
                if not updates: raise RuntimeError('No official optimizer updates recorded')
                if smoke:
                    if not any(not torch.equal(before[k],v) for k,v in model.pre_projection.named_parameters()):
                        raise RuntimeError('Smoke did not change projection weights')
                    pd = VisADataset(items[:2],args.dtd_root,training=False,
                                     distribution_mode=table[c]['explicit_distribution_mode'])
                    images = torch.stack([pd[i]['image'] for i in range(len(pd))])
                    with meter.measure('smoke_inference_before_reload',c):
                        scores, maps = model._predict(images)
                    del model; gc.collect(); torch.cuda.empty_cache()
                    with meter.measure('strict_smoke_reload',c):
                        model = model_for(args.backbone_weights,device)
                        load_last(model,checkpoint,c,require_formal=False,protocol_id=PROTOCOL)
                    with meter.measure('smoke_inference_after_reload',c):
                        scores2, maps2 = model._predict(images)
                    np.testing.assert_array_equal(scores,scores2); np.testing.assert_array_equal(maps,maps2)
                    for i,(score,value) in enumerate(zip(scores2,maps2)):
                        path = target/'smoke_maps'/f'{i:06d}.npz'
                        npz_write(path, input_map=np.asarray(value,np.float32),image_score=np.asarray(score,np.float64))
                        with np.load(path,allow_pickle=False) as cached:
                            np.testing.assert_array_equal(cached['input_map'],np.asarray(value,np.float32))
                            assert float(cached['image_score']) == float(score)
                    json_write(target/'smoke_verification.json',dict(real_optimizer_update=True,
                        strict_reload_prediction_equal=True, FP32_save_load_equal=True,
                        test_metrics_used=False, train_normal_images=len(items), optimizer_updates=updates))
                json_write(target/'complete.json',dict(status='complete',protocol_id=PROTOCOL,category=c,
                    formal_model=not smoke, completed_epochs=1 if smoke else 640, checkpoint=checkpoint.name,
                    branch=table[c], optimizer_updates=updates, foreground_enabled=ds.class_fg,
                    foreground_observation='main-process path reads only; workers>0 reads happen in worker processes'))
                print(f'train visa/{c} complete; smoke={smoke}; updates={updates}',flush=True)
                del model, dl, ds; gc.collect(); torch.cuda.empty_cache()
        complete(out,args,formal_model=not smoke)


def saved_inputs(target, category):
    """Read native and mapped FP32 predictions plus associated cached original-GT masks."""
    import numpy as np
    state = read_json(target/'complete.json')
    if state.get('protocol_id') != PROTOCOL or state.get('status') != 'complete' or not state.get('formal_model'):
        raise ValueError('Complete formal VisA predictions required for cache evaluation')
    rows = read_json(target/'samples.json')
    if not rows or len(rows) != state['image_count'] or len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Missing/duplicate cache records')
    maps,masks = [],[]
    for row in rows:
        if row['category'] != category or not np.isfinite(row['image_score']):
            raise ValueError('Wrong cached category or nonfinite official image score')
        rel = Path(row['prediction_file'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Invalid prediction cache path')
        with np.load(target/rel,allow_pickle=False) as d:
            if str(d['relative_path']) != row['relative_path'] or str(d['category']) != category or int(d['label']) != row['label'] or float(d['image_score']) != row['image_score']:
                raise ValueError('Cached prediction identity mismatch')
            native, mapped, mask = d['input_map'],d['evaluation_map'],d['mask']
            if native.dtype != np.float32 or mapped.dtype != np.float32 or native.shape != (288,288) or mapped.shape != tuple(row['evaluation_hw']) or mask.dtype != np.bool_ or mask.shape != mapped.shape or not np.isfinite(native).all() or not np.isfinite(mapped).all():
                raise ValueError('Invalid cached FP32 map or GT geometry')
            maps.append(mapped); masks.append(mask)
    return rows,maps,masks


def evaluate(args):
    with operation(args) as out:
        torch = require_runtime(out); device = device_for(args.device)
        from external_baselines.glass_mvtec_pretrained.runner import Resources
        from external_baselines.patchcore_official_eval.evaluation import calculate
        import numpy as np
        metrics, operating = [],[]
        if args.trained_root:
            if not args.dataset_root or not args.backbone_weights:
                raise ValueError('New inference requires --dataset-root and --backbone-weights')
            from .data import records, metadata, geometry, map_to_evaluation, VisADataset, evaluation_mask
            from external_baselines.glass_external.adapter import model_for, load_last
            from external_baselines.glass_external.official import modules
            root,_,test,counts = records('visa',args.dataset_root,args.categories)
            csv_write(out/'data_counts.csv',counts)
            for row in counts:
                if (row['test_normal'],row['test_anomaly']) != COUNTS[row['category']][1:]:
                    raise ValueError('Formal evaluation test counts mismatch; inspect data_counts.csv')
            weight = weight_identity(args.backbone_weights)
            cfg = read_json(args.trained_root/'config.json')
            if cfg.get('protocol_id') != PROTOCOL or not set(args.categories).issubset(cfg['categories']):
                raise ValueError('Matching VisA training configuration/categories required')
            if cfg['backbone_weight']['sha256'] != weight['sha256']:
                raise ValueError('Training/inference encoder weight mismatch')
            json_write(out/'config.json',dict(cfg,evaluation_device=str(device),inference_batch=args.inference_batch))
            if args.inference_batch < 1: raise ValueError('inference-batch must be positive')
        else:
            cfg = read_json(args.prediction_root/'config.json')
            if cfg['protocol_id'] != PROTOCOL: raise ValueError('Wrong prediction protocol')
            json_write(out/'config.json',dict(cfg,cache_reevaluation=True,evaluation_device=str(device)))
        for c in args.categories:
            target = out/c; target.mkdir(); meter = Resources(target,device)
            if args.trained_root:
                tc = args.trained_root/c
                state = read_json(tc/'complete.json')
                if state.get('status') != 'complete' or state.get('protocol_id') != PROTOCOL or state.get('category') != c or not state.get('formal_model') or state.get('completed_epochs') != 640:
                    raise ValueError('Formal 640th epoch category checkpoint required')
                branch = state['branch']
                if branch != cfg['branches'][c]: raise ValueError('Checkpoint branch metadata mismatch')
                modules().utils.fix_seeds(0)
                with logging(target/'inference.log'):
                    with meter.measure('strict_fixed_final_model_load',c):
                        model = model_for(args.backbone_weights,device)
                        loaded = load_last(model,tc/'ckpt_last.pth',c,protocol_id=PROTOCOL)
                        if loaded['distribution'] != branch['excel_distribution']: raise ValueError('Checkpoint/Excel branch mismatch')
                    ds = VisADataset(test[c],'',training=False,distribution_mode=branch['explicit_distribution_mode'])
                    loader = torch.utils.data.DataLoader(ds,batch_size=args.inference_batch,shuffle=False,
                                                        num_workers=args.workers,pin_memory=True)
                    rows = []; cursor = 0
                    with meter.measure('official_inference_and_FP32_save',c):
                        for batch in loader:
                            scores, native_maps = model._predict(batch['image'])
                            for image_path,score,native in zip(batch['image_path'],scores,native_maps):
                                record = test[c][cursor]
                                if str(Path(image_path).resolve()) != str(Path(record.path).resolve()):
                                    raise ValueError('Dataset/image record order mismatch')
                                geom = geometry(record.path)
                                row = metadata(record,root,'test')
                                row.update(input_hw=[288,288],geometry=geom,
                                    coordinate_rule='restore centered crop; exterior boundary extension; original H//4,W//4',
                                    prediction_file=f'maps/{cursor:06d}.npz',image_score=float(score))
                                native = np.asarray(native,np.float32); mapped = map_to_evaluation(native,geom)
                                mask = evaluation_mask(row,root)
                                npz_write(target/row['prediction_file'], input_map=native,evaluation_map=mapped,mask=mask,
                                    relative_path=np.asarray(row['relative_path']), category=np.asarray(c),label=np.asarray(row['label']),
                                    image_score=np.asarray(row['image_score'],np.float64))
                                rows.append(row); cursor += 1
                            json_write(target/'samples.json',rows)
                            print(f'predict visa/{c}: {cursor}/{len(test[c])}',flush=True)
                    if cursor != len(test[c]): raise RuntimeError('Incomplete inference')
                    csv_write(target/'image_scores.csv',rows)
                    json_write(target/'complete.json',dict(status='complete',protocol_id=PROTOCOL,
                        formal_model=True,category=c,image_count=len(rows),branch=branch, native_map_hw=[288,288]))
                    del model,loader,ds; gc.collect(); torch.cuda.empty_cache()
                cache = target
            else:
                cache = args.prediction_root/c
            with meter.measure('saved_prediction_and_GT_read',c):
                rows,maps,masks = saved_inputs(cache,c)
            with meter.measure('fast_CUDA_metrics_and_fixed_FPR',c):
                values,fixed = calculate(rows,maps,masks,str(device),allow_cpu=False)
            result = dict(category=c, method=METHOD, **values)
            metrics.append(result); operating.extend(dict(category=c,method=METHOD,**r) for r in fixed)
            csv_write(target/'category_metrics.csv',[result]); csv_write(target/'fixed_fpr.csv',[dict(category=c,method=METHOD,**r) for r in fixed])
            csv_write(out/'category_metrics.csv',metrics); csv_write(out/'fixed_fpr.csv',operating)
            json_write(target/'evaluation_complete.json',dict(status='complete',protocol_id=PROTOCOL,category=c,
                prediction_root=str(cache.resolve()), image_count=len(rows), formal_cuda_metrics=True))
            del maps,masks; gc.collect(); torch.cuda.empty_cache()
        write_summary(out,metrics,operating,args.categories)
        complete(out,args,formal_model=True,formal_cuda_metrics=True)


def write_summary(out, metrics, operating, categories):
    from external_baselines.patchcore_official_eval.evaluation import summarize as shared_summary
    macro,fmacro = shared_summary(metrics,operating,categories,'visa')
    for row in macro+fmacro: row.update(method=METHOD,protocol_id=PROTOCOL)
    csv_write(out/'macro_metrics.csv',macro); csv_write(out/'fixed_fpr_macro.csv',fmacro)


def summarize(args):
    import csv
    with operation(args) as out:
        metrics, operating = [],[]
        for root in args.result_roots:
            state = read_json(root/'complete.json')
            if state.get('protocol_id') != PROTOCOL or state.get('status') != 'complete' or not state.get('formal_cuda_metrics'):
                raise ValueError('Only complete matching formal CUDA evaluations can be merged')
            for filename,destination in (('category_metrics.csv',metrics),('fixed_fpr.csv',operating)):
                with (root/filename).open(newline='',encoding='utf-8') as handle:
                    for row in csv.DictReader(handle):
                        if row['category'] not in args.categories: continue
                        for k,v in row.items():
                            if k.endswith(('count','count_sum')) or k in ('region_count','small_region_count','small_region_image_count'):
                                row[k] = int(v)
                            elif k in ('image_AUROC','image_AP','pixel_AUROC','pixel_AP','AUPRO_at_0p3','fpr_cap',
                                       'actual_fpr','defect_pixel_recall','region_mean_coverage','small_region_mean_coverage'):
                                row[k] = None if v in ('','N/A','None') else float(v)
                        destination.append(row)
        write_summary(out,metrics,operating,args.categories)
        csv_write(out/'category_metrics.csv',metrics); csv_write(out/'fixed_fpr.csv',operating)
        json_write(out/'sources.json',dict(protocol_id=PROTOCOL,result_roots=[str(p.resolve()) for p in args.result_roots]))
        complete(out,args,formal_cuda_metrics=True)
