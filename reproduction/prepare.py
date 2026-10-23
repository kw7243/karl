"""Read-only source/data checks and released checkpoint preparation; no evaluation."""
import argparse, ast, hashlib, json, os, platform, subprocess, time, urllib.request
from pathlib import Path
import torch

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output)
r=Path.cwd()
assets=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-assets/checkpoints')
tree=ast.parse(Path('kolmogorov_tokenizers/pretrained_models/download.py').read_text())
urls=ast.literal_eval(next(x.value for x in tree.body if isinstance(x,ast.Assign)))
receipts=[]
for dataset in ['imagenet','imagenet100']:
    name='karl_small_vqgan_quantized_latents.pth'
    dest=assets/dataset/name;dest.parent.mkdir(parents=True,exist_ok=True)
    url=urls[dataset][name].replace('dl=0','dl=1')
    if not dest.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'Wget/1.16 (linux-gnu)'})
        with urllib.request.urlopen(req,timeout=120) as response,dest.with_suffix('.partial').open('xb') as handle:
            assert response.status==200
            while True:
                data=response.read(8*1024*1024)
                if not data:break
                handle.write(data)
        dest.with_suffix('.partial').rename(dest)
    print('CHECKPOINT',dataset,dest.stat().st_size,flush=True)
    checkpoint=torch.load(dest,map_location='cpu',mmap=True)
    receipt={'dataset':dataset,'path':str(dest),'source_url':url,'bytes':dest.stat().st_size,'sha256':sha(dest),'keys':list(checkpoint)}
    for key,value in checkpoint.items():
        if key in ['args','epoch','version']: receipt[key]=str(value)
        if isinstance(value,dict):
            receipt[key+'_summary']={'keys':len(value),'parameters':sum(x.numel() for x in value.values() if isinstance(x,torch.Tensor)),'sample_keys':list(value)[:10]}
            if key in ['model','ema','state_dict']:
                shapes={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in value.items() if isinstance(v,torch.Tensor)}
                (out/(dataset+'-'+key+'-shapes.json')).write_text(json.dumps(shapes,indent=2))
    receipts.append(receipt)
    (out/'checkpoints.json').write_text(json.dumps(receipts,indent=2));print(json.dumps(receipt),flush=True)
    del checkpoint
# Hash the exact paper split and preserve an independent label-map identity.
val=Path('/data/vision/torralba/datasets/imagenet100/val')
synsets=ast.literal_eval(next(x.value for x in ast.parse(Path('run_scripts/create_imagenet100.py').read_text()).body if isinstance(x,ast.Assign)))
classes=sorted(x.name for x in val.iterdir());assert sorted(synsets)==classes
manifest=[]
for label,synset in enumerate(classes):
    images=sorted(x for x in (val/synset).iterdir() if x.suffix.lower() in ['.jpeg','.jpg','.png'])
    assert len(images)==50
    for img in images:manifest.append({'relative_path':str(img.relative_to(val)),'resolved_path':str(img.resolve()),'label':label,'bytes':img.stat().st_size,'sha256':sha(img)})
assert len(manifest)==5000
(out/'dataset-manifest.json').write_text(json.dumps({'root':str(val),'class_to_idx':dict(zip(classes,range(100))),'samples':manifest},indent=2))
info={'job_id':os.getenv('SLURM_JOB_ID'),'stage':str(r),'node':platform.node(),'python':platform.python_version(),'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'checkpoint_manifest_sha256':sha(out/'checkpoints.json'),'dataset_manifest_sha256':sha(out/'dataset-manifest.json')}
(out/'preparation.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
