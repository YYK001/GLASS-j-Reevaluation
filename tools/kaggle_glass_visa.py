"""One-cell Kaggle setup for the standalone GLASS repository. Not executed locally.

Default: prepare resources, validate all 12 categories, run tests and one real smoke.
--formal additionally trains/evaluates the selected disjoint category groups.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import zipfile

REMOTE = 'https://github.com/YYK001/GLASS-j-Reevaluation.git'
MOD = 'external_baselines.glass_visa'
CATEGORIES = ('candle','capsules','cashew','chewinggum','fryum','macaroni1',
              'macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')
DTD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
FG_ID = '1Fn84QCfMtgBGEDcmY44v97Ci8wwpABK8'


def extract_foreground(archive, target):
    """Extract only VisA masks from the official foreground release, unchanged.

    Preserve category/filename; record the original archive path. Reject aliases,
    duplicate destinations and nonregular TAR files rather than guessing a layout.
    """
    target = Path(target); target.mkdir(parents=True,exist_ok=False)
    rows, seen = [],set()
    def destination(name, regular=True):
        p = PurePosixPath(name)
        if p.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
            return None
        matches = [c for c in CATEGORIES if c in p.parts[:-1] or 'visa_'+c in p.parts[:-1]]
        if not matches: return None
        if len(matches) != 1 or p.is_absolute() or '..' in p.parts or not regular:
            raise ValueError('Unsafe/ambiguous official foreground archive member: '+name)
        relative = Path(matches[0])/p.name
        if relative.as_posix() in seen:
            raise ValueError('Duplicate foreground destination: '+str(relative))
        seen.add(relative.as_posix())
        result = target/relative; result.parent.mkdir(parents=True,exist_ok=True)
        rows.append(dict(archive_member=name,category=matches[0],foreground_file=relative.as_posix()))
        return result
    def copy(source, dest):
        with source, dest.open('xb') as output: shutil.copyfileobj(source,output,1024*1024)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.is_dir(): continue
                dest = destination(m.filename)
                if dest: copy(z.open(m),dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as z:
            for m in z:
                dest = destination(m.name,m.isfile())
                if dest: copy(z.extractfile(m),dest)
    else:
        raise ValueError('Official foreground download is not a ZIP/TAR archive; keep it for inspection')
    (target/'archive_paths.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    missing = [c for c in CATEGORIES if not (target/c).is_dir()]
    if missing: raise ValueError('Official archive layout did not resolve VisA categories: '+str(missing))
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/kaggle/working/GLASS-j-Reevaluation'))
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--foreground-root',type=Path)
    p.add_argument('--dtd-root',type=Path)
    p.add_argument('--formal',action='store_true')
    p.add_argument('--gpu0',nargs='+',default=['candle'])
    p.add_argument('--gpu1',nargs='+',default=['capsules'])
    args = p.parse_args(argv)
    chosen = args.gpu0+args.gpu1
    if len(chosen) != len(set(chosen)) or not set(chosen).issubset(CATEGORIES):
        raise ValueError('GPU groups must contain distinct official categories')
    work = Path('/kaggle/working'); repo = args.repo.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    assets = work/'glass_visa_assets'; assets.mkdir(parents=True,exist_ok=True)
    output = work/f'glass_visa_{stamp}'; output.mkdir(exist_ok=False)
    py = work/'glass-visa-venv/bin/python'
    env = os.environ.copy()
    env.pop('UV_SYSTEM',None)
    env.update(UV_PYTHON_INSTALL_DIR=str(work/'glass_visa_python'),UV_NO_CACHE='1',
               UV_LINK_MODE='copy',UV_HTTP_TIMEOUT='120',PYTHONUNBUFFERED='1')

    def run(command, log_name, cwd=repo):
        command = list(map(str,command))
        print('\n>>> '+' '.join(command),flush=True)
        with (output/(log_name+'.log')).open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); log.flush()
                    print('['+log_name+'] '+line,end='',flush=True)
                code = process.wait()
        print(f'[{log_name}] exit={code}',flush=True)
        if code: raise RuntimeError(f'{log_name} failed; exit={code}; see {output}')

    def entry(stage,log_name,*extra):
        run([py,'-u','-m',MOD,stage,*extra],log_name)

    remote = subprocess.check_output(['git','remote','get-url','origin'],cwd=repo,text=True).strip()
    assert remote == REMOTE, ('Unexpected Git remote',remote)
    assert (repo/'external_baselines/glass_visa/__main__.py').is_file(), 'GitHub code is missing VisA support'
    print('Outputs:',output,'\nFormal:',args.formal,'\nGPU groups:',args.gpu0,args.gpu1,flush=True)
    uv = shutil.which('uv'); assert uv, 'Kaggle uv is required to prepare independent Python3.10'
    if not py.is_file():
        run([uv,'python','install','3.10'],'python_prepare')
        run([uv,'venv','--python','3.10','--seed',py.parent.parent],'venv_prepare')
    run([uv,'pip','install','--python',py,'torch==2.1.2','torchvision==0.16.2',
         '--index-url','https://download.pytorch.org/whl/cu118'],'torch_prepare')
    run([uv,'pip','install','--python',py,'-r',repo/'requirements.txt'],'dependencies_prepare')
    run([py,'-m','pip','check'],'pip_check')
    run([py,'-u','-c','''
import torch
print('torch:',torch.__version__,'CUDA:',torch.version.cuda)
assert torch.cuda.is_available() and torch.cuda.device_count()==2, 'Enable GPU T4 x2 in Notebook settings'
for gpu in (0,1):
    with torch.cuda.device(gpu):
        x=torch.ones(4,device=f'cuda:{gpu}'); assert x.sum().item()==4
        free,total=torch.cuda.mem_get_info()
    print(f'GPU{gpu}: {torch.cuda.get_device_name(gpu)}; free={free/1024**3:.2f} GiB')
    assert 'T4' in torch.cuda.get_device_name(gpu), 'This cell targets dual Tesla T4'
'''],'gpu_check')
    visa = args.dataset_root
    if visa is None:
        candidates = sorted({f.parent.parent for f in Path('/kaggle/input').rglob('1cls.csv')
                             if f.parent.name=='split_csv' and (f.parent.parent/'candle').is_dir()})
        assert len(candidates)==1, f'Need one mounted VisA raw dataset or explicit --dataset-root: {candidates}'
        visa = candidates[0]
    assert (visa/'split_csv/1cls.csv').is_file(), 'Mount the raw VisA dataset with split_csv/1cls.csv'
    weight = assets/'wide_resnet50_2-95faca4d.pth'
    if not weight.is_file():
        part = weight.with_suffix('.pth.part')
        run([py,'-u','-c',"import torch,sys; torch.hub.download_url_to_file('https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',sys.argv[1],hash_prefix='95faca4d')",part],'weight_download')
        part.replace(weight)
    dtd = args.dtd_root
    if dtd is None:
        candidates = [assets/'dtd/images',work/'destseg_assets/dtd/images']
        candidates += [r/'images' for r in Path('/kaggle/input').rglob('dtd') if r.is_dir()]
        dtd = next((r for r in candidates if r.is_dir() and len(list(r.glob('*/*.jpg')))==5640),None)
    if dtd is None:
        archive = assets/'dtd-r1.0.1.tar.gz'
        if not archive.is_file():
            run(['curl','-fL','--retry','3',DTD_URL,'-o',str(archive)+'.part'],'dtd_download')
            Path(str(archive)+'.part').replace(archive)
        with tarfile.open(archive) as z:
            images = [m for m in z if m.isfile() and len(PurePosixPath(m.name).parts)==4
                      and PurePosixPath(m.name).parts[:2]==('dtd','images') and m.name.lower().endswith('.jpg')]
            assert len(images)==5640 and len({m.name for m in images})==5640, f'Incomplete official DTD archive: {len(images)} images'
            assert sum(m.size for m in images) < shutil.disk_usage(work).free, 'Insufficient disk for DTD extraction'
            for i,m in enumerate(images,1):
                assert '..' not in PurePosixPath(m.name).parts, 'Unsafe DTD archive member'
                dest = assets/Path(*PurePosixPath(m.name).parts)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    assert dest.stat().st_size==m.size, f'Existing incomplete DTD file: {dest}'
                    continue
                with z.extractfile(m) as src,dest.open('xb') as dst: shutil.copyfileobj(src,dst)
                if i%1000==0 or i==len(images): print('DTD:',i,'/',len(images),flush=True)
        archive.unlink()  # Only the downloaded resource archive; extracted DTD is retained.
        dtd = assets/'dtd/images'
    foreground = args.foreground_root
    if foreground is None:
        candidates = [visa/'fg_mask',assets/'fg_mask']+list(assets.glob('foreground_*'))
        candidates += list(Path('/kaggle/input').rglob('fg_mask'))
        foreground = next((r for r in candidates if all((r/c).is_dir() for c in CATEGORIES)),None)
    if foreground is None:
        archive = assets/'official_foreground_archive'
        if not archive.is_file():
            run([py,'-m','gdown','--id',FG_ID,'-O',str(archive)+'.part'],'foreground_download')
            Path(str(archive)+'.part').replace(archive)
        # Only extract VisA masks; never copy the main dataset or all other foreground datasets.
        foreground = extract_foreground(archive,assets/f'foreground_{stamp}')
        archive.unlink()  # Official mask files and archive path record are retained.
    print('VISA_ROOT =',visa,'\nFG_ROOT =',foreground,'\nDTD_ROOT =',dtd,
          '\nFree disk GiB:',round(shutil.disk_usage(work).free/1024**3,2),flush=True)
    (output/'paths.json').write_text(json.dumps(dict(repo=str(repo),python=str(py),visa_root=str(visa),
        foreground_root=str(foreground),dtd_root=str(dtd),backbone=str(weight),formal=args.formal,
        gpu0=args.gpu0,gpu1=args.gpu1),indent=2),encoding='utf-8')
    common = ['--dataset-root',visa,'--dtd-root',dtd,'--foreground-root',foreground,'--backbone-weights',weight]
    entry('check','check',*common,'--categories','all','--output-dir',output/'check')
    run([py,'-m','unittest','discover','-s','external_baselines/glass_visa/tests','-v'],'visa_tests')
    run([py,'-m','pytest','-q','DINOv3/tests/test_glass_mvtec_pretrained.py'],'mvtec_regression')
    entry('smoke','smoke',*common,'--categories','candle','--device','cuda:0','--workers','0',
          '--smoke-images','8','--output-dir',output/'smoke')
    if args.formal:
        assert shutil.disk_usage(work).free > 3*1024**3, 'Keep at least 3 GiB free before the selected batch'
        def worker(gpu,categories):
            train,evaldir = output/f'train_gpu{gpu}',output/f'eval_gpu{gpu}'
            entry('train',f'train_gpu{gpu}',*common,'--categories',*categories,'--device',f'cuda:{gpu}',
                  '--workers','2','--output-dir',train)
            entry('evaluate',f'eval_gpu{gpu}','--trained-root',train,'--dataset-root',visa,
                  '--backbone-weights',weight,'--categories',*categories,'--device',f'cuda:{gpu}',
                  '--workers','2','--inference-batch','1','--output-dir',evaldir)
            return evaldir
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker,0,args.gpu0),pool.submit(worker,1,args.gpu1)]
            roots = [f.result() for f in futures]
        entry('summarize','summarize','--result-roots',*roots,'--categories',*chosen,'--output-dir',output/'summary')
    report = work/f'GLASS_visa_{stamp}_text.zip'
    with zipfile.ZipFile(report,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in output.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.csv','.log'):
                z.write(f,str(Path(output.name)/f.relative_to(output)))
    print('\nCompleted. Formal training started:',args.formal,'\nResults:',output,'\nText report:',report,
          '\nWeights and native/mapped continuous predictions are retained separately.',flush=True)


if __name__=='__main__': main()
