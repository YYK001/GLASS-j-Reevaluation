"""Small synthetic checks; real official modules, no released weights/full-data evaluation."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from external_baselines.glass_mvtec_pretrained import checkpoints as cp, data, runner
from external_baselines.glass_external import adapter
from external_baselines.glass_external.official import modules
from DINOv3.tests.test_glass_btad_external import small_model


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path/'mvtec'
    a = np.arange(400*500, dtype=np.uint32).reshape(400,500)
    for c in ('bottle','toothbrush','wood'):
        for sub in ('train/good','test/good','test/crack','ground_truth/crack'):
            (root/c/sub).mkdir(parents=True)
        for sub in ('train/good','test/good','test/crack'):
            Image.fromarray((a%256).astype(np.uint8)).save(root/c/sub/'000.png')
        mask=np.zeros((400,500),np.uint8);mask[:2,:2]=255  # outside crop, image remains abnormal
        Image.fromarray(mask).save(root/c/'ground_truth/crack/000_mask.png')
    return root


def model():
    m=small_model()
    m.anomaly_segmentor.target_size=(288,288)
    return m


def save_heads(m,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'discriminator':m.discriminator.state_dict(),'pre_projection':m.pre_projection.state_dict()},path)


@pytest.mark.parametrize('category,resize', [('bottle',288),('toothbrush',329),('wood',329)])
def test_official_transform_and_actual_crop(inputs,category,resize):
    root,tests,_=data.records(inputs,(category,))
    items=tests[category]
    transform,actual=data.official_transform(items)
    assert actual==resize
    official=modules().mvtec.MVTecDataset(str(root),classname=category,
        split=modules().mvtec.DatasetSplit.TEST,resize=288,imagesize=288,distribution=0,fg=0)
    x,geom=data.load_rgb(items[0],transform,actual)
    with Image.open(items[0].path) as im:
        torch.manual_seed(0); expected=official.transform_img(im.convert('RGB'))
    assert torch.equal(x,expected)
    assert geom['crop_top']==round((geom['resized_height']-288)/2)
    assert geom['crop_left']==round((geom['resized_width']-288)/2)
    assert x.shape==(3,288,288)
    # Synthetic coordinate map exercises the exact same integer crop.
    field=np.arange(geom['resized_height']*geom['resized_width'],dtype=np.float32).reshape(
        geom['resized_height'],geom['resized_width'])
    expected=TF.center_crop(torch.from_numpy(field)[None], [288,288])[0].numpy()
    np.testing.assert_array_equal(data.reference_crop(field,geom),expected)
    bad=next(r for r in items if r.label==1)
    mask,audit=data.crop_gt(bad,geom)
    assert mask.shape==(288,288) and set(np.unique(mask))<={0,1}
    assert audit['cropped_empty'] and not audit['original_empty'] and audit['image_label']==1


def test_strict_published_loading_including_bn_buffers(tmp_path):
    m=model(); path=tmp_path/'ckpt_best_543.pth';save_heads(m,path)
    cp.load_released(m,path)
    assert not m.training and not any(p.requires_grad for p in m.parameters())
    d=cp.describe(path,'bottle');assert len(d['sha256'])==64 and d['filename_epoch_index']==543
    assert d['backbone_state'].startswith('external')
    original=cp.read_state(path)
    for mutate in ('projection','bn','shape','config'):
        state={k:dict(v) for k,v in original.items()}
        if mutate=='projection':state.pop('pre_projection')
        elif mutate=='bn':state['discriminator'].pop(next(k for k in state['discriminator'] if 'num_batches_tracked' in k))
        elif mutate=='shape':
            key=next(iter(state['pre_projection']));state['pre_projection'][key]=torch.zeros(1)
        else:state['config']={'imagesize':256}
        torch.save(state,path)
        with pytest.raises(ValueError):cp.load_released(m,path)


def test_release_mapping_never_selects_first_or_best_locally(tmp_path):
    assert set(cp.RELEASE['entries'])==set(data.CATEGORIES) and len(data.CATEGORIES)==15
    path=tmp_path/'results/models/backbone_0/mvtec_bottle'/cp.entry('bottle')['filename']
    path.parent.mkdir(parents=True);path.touch()
    assert cp.resolve(tmp_path,'bottle')==path
    extra=path.with_name('ckpt_best_999.pth');extra.touch()
    with pytest.raises(ValueError,match='additional'):cp.resolve(tmp_path,'bottle')
    with pytest.raises(FileNotFoundError):cp.resolve(tmp_path,'wood')


def test_wrapper_matches_real_official_predict_and_roundtrip(tmp_path):
    m=model();path=tmp_path/'ckpt_best_543.pth';save_heads(m,path)
    restored=model();cp.load_released(restored,path)
    image=torch.rand(2,3,64,64)
    expected=m._predict(image)
    scores,maps=runner.infer(restored,image)
    np.testing.assert_array_equal(scores,np.asarray(expected[0]))
    np.testing.assert_array_equal(maps,np.asarray(expected[1]))
    file=tmp_path/'saved.npz';runner._atomic_npz(file,anomaly_map=maps,image_score=scores)
    with np.load(file) as z:
        np.testing.assert_array_equal(z['anomaly_map'],maps)
        np.testing.assert_array_equal(z['image_score'],scores)


def test_predict_saved_gt_and_offline_evaluate_compare(inputs,tmp_path,monkeypatch):
    weights=tmp_path/'weights';path=weights/'mvtec_bottle'/cp.entry('bottle')['filename']
    save_heads(model(),path)
    monkeypatch.setattr(adapter,'model_for',lambda *a,**k:model())
    def forbidden(*a,**k):raise AssertionError('training/tester must never run')
    monkeypatch.setattr(modules().glass.GLASS,'trainer',forbidden)
    monkeypatch.setattr(modules().glass.GLASS,'tester',forbidden)
    backbone=tmp_path/'backbone.pth';backbone.write_bytes(b'synthetic path only')
    out=tmp_path/'prediction'
    runner.main(['predict','--category','bottle','--dataset-root',str(inputs),
        '--weights-root',str(weights),'--backbone-weights',str(backbone),
        '--device','cpu','--batch-size','1','--output-dir',str(out)])
    state=json.loads((out/'complete.json').read_text(encoding='utf-8'));assert not state['all_15_completed']
    root,rows=runner.saved_inputs(out,('bottle',));assert len(rows)==2
    assert sorted(int(r['image_label']) for r in rows)==[0,1]
    for r in rows:runner.load_sample(root,r)
    # Evaluate must not construct any model or read source RGB / source GT.
    monkeypatch.setattr(adapter,'model_for',forbidden)
    monkeypatch.setattr(runner,'device_for',lambda *a,**k:torch.device('cpu'))
    def fake_metrics(labels,masks,maps,**kw):
        assert kw['allow_cpu_fallback'] is False
        return {k:0.123 for k in runner.METRIC_NAMES}
    monkeypatch.setattr(runner,'evaluate_fast',fake_metrics)
    evaluation=tmp_path/'evaluation'
    runner.main(['evaluate','--category','bottle','--prediction-root',str(out),'--output-dir',str(evaluation)])
    metric=runner._read_csv(evaluation/'category_metrics.csv')[0]
    expected=runner.image_metrics([int(r['image_label']) for r in rows],[float(r['image_score']) for r in rows])
    assert float(metric['image_AUROC'])==expected['image_AUROC']
    assert float(metric['image_AUPR'])==expected['image_AUPR']
    assert runner._read_csv(evaluation/'macro_metrics.csv')[0]['category_count']=='1'
    index=[]
    for i,r in enumerate(rows):
        p=tmp_path/f'ref{i}.npy';np.save(p,np.full((100,125),i,dtype=np.float32))
        index.append(dict(category='bottle',relative_path=r['relative_path'],map_path=str(p),
            map_key='',image_score=i,coordinate_system='full_frame'))
    csvpath=tmp_path/'ref.csv';runner._write_csv(csvpath,index)
    comparison=tmp_path/'comparison'
    runner.main(['compare','--category','bottle','--prediction-root',str(out),'--reference-index',str(csvpath),
                 '--reference-method','EXTERNAL','--output-dir',str(comparison)])
    assert json.loads((comparison/'complete.json').read_text(encoding='utf-8'))['compare_executed']
    assert all(r['comparison']=='REFERENCE_EXTERNAL_minus_GLASS_J_RELEASED'
               for r in runner._read_csv(comparison/'common_fov_differences.csv'))
    runner._write_csv(out/'sample_scores.csv',rows[:1])
    with pytest.raises(ValueError,match='missing saved samples'):runner.saved_inputs(out,('bottle',))


def test_summary_equal_categories_and_na():
    metrics=[dict(category=c,method=cp.METHOD,**{k:v for k in runner.METRIC_NAMES}) for c,v in [('bottle',.2),('wood',.8)]]
    fpr=[dict(category=c,fpr_cap=cap,actual_fpr=cap,defect_pixel_recall=v,region_mean_coverage=v,
        small_region_mean_coverage='N/A' if c=='bottle' else .7,region_count=n,small_region_count=0 if c=='bottle' else 2,
        small_region_image_count=0 if c=='bottle' else 1)
        for c,v,n in [('bottle',.2,100),('wood',.8,2)] for cap in (.01,.05)]
    macro,f=runner.summarize(metrics,fpr,('bottle','wood'))
    assert macro[0]['pixel_AUPR']==.5 and macro[0]['category_count']==2
    assert f[0]['small_region_mean_coverage']==.7 and f[0]['small_region_mean_coverage_valid_category_count']==1
    with pytest.raises(ValueError):runner.summarize(metrics[:1],fpr,('bottle','wood'))


def test_reference_key_and_coordinate_validation(tmp_path):
    path=tmp_path/'reference.csv';wanted=[dict(category='bottle',relative_path='bottle/test/good/000.png')]
    args=SimpleNamespace(reference_index=str(path))
    row=dict(**wanted[0],map_path='map.npy',map_key='',image_score=1,coordinate_system='cropped')
    runner._write_csv(path,[row])
    with pytest.raises(ValueError,match='full_frame'):runner.reference_index(args,wanted)
    row['coordinate_system']='full_frame';runner._write_csv(path,[row,row])
    with pytest.raises(ValueError,match='duplicate'):runner.reference_index(args,wanted)


def test_download_pinned_ids_and_manual_fallback(tmp_path,monkeypatch):
    import sys
    calls=[]
    def fetch(**kwargs):
        calls.append(kwargs['id']);torch.save({'discriminator':{},'pre_projection':{}},kwargs['output']);return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=fetch))
    cp.download(tmp_path/'release',('bottle',))
    assert calls==[cp.entry('bottle')['file_id']]
    assert cp.resolve(tmp_path/'release','bottle').is_file()
    with pytest.raises(FileExistsError):cp.download(tmp_path/'release',('bottle',))


def test_cli_independent_help():
    for args in (['--help'],)+tuple([cmd,'--help'] for cmd in ('download','check','predict','evaluate','compare')):
        with pytest.raises(SystemExit) as result:runner.build_parser().parse_args(args)
        assert result.value.code==0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable; official WRN small check not run')
def test_real_official_wrn_cuda_single_batch(tmp_path):
    # Official WRN architecture + 1536 heads, random backbone, synthetic 64px input.
    # This is not validation of ImageNet weights or real MVTec inference.
    torch.set_num_threads(1)
    m=small_model('cuda:0',real_backbone=True)
    m.anomaly_segmentor.target_size=(288,288)
    path=tmp_path/'ckpt_best_543.pth';save_heads(m,path)
    cp.load_released(m,path)
    images=torch.rand(1,3,64,64,device='cuda:0')
    original=m._predict(images)
    scores,maps=runner.infer(m,images)
    np.testing.assert_array_equal(scores,np.asarray(original[0]))
    np.testing.assert_array_equal(maps,np.asarray(original[1]))
    del m,images;torch.cuda.empty_cache()
