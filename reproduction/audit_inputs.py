"""CPU-only file integrity and image-decode audit; no model inference or scoring."""
import hashlib,json,os,platform,sys,time
from pathlib import Path
from PIL import Image
import torch
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
start=time.perf_counter()
root=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023')
preparation=root/'karl-reproduction-results/cpu-prepare-2826204'
manifest=json.loads((preparation/'dataset-manifest.json').read_text())
assert len(manifest['samples'])==5000
assert len({Path(x['relative_path']).stem for x in manifest['samples']})==5000
modes={};formats={};small=0
for sample in manifest['samples']:
    path=Path(manifest['root'])/sample['relative_path']
    with Image.open(path) as image:
        image.load();image.convert('RGB')
        modes[image.mode]=modes.get(image.mode,0)+1
        formats[image.format]=formats.get(image.format,0)+1
        small+=int(min(image.size)<256)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
metric_roots=[root/'karl-reproduction-assets/dreamsim',Path('/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/torch/hub/checkpoints')]
files=[p for folder in metric_roots for p in folder.rglob('*') if p.is_file()]
files += [root/'karl-reproduction-assets/dino-base-pretrained.pth',Path('/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/lib/python3.10/site-packages/lpips/weights/v0.1/alex.pth')]
weights=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))]
record={'run_id':out.name,'node_id':'SETUP_READY','status':'COMPLETED','job_id':os.getenv('SLURM_JOB_ID'),'node':platform.node(),'stage':str(Path.cwd()),'sample_count':5000,'all_images_decode':True,'unique_output_stems':5000,'modes':modes,'formats':formats,'images_smaller_than_256':small,'small_image_handling':'released CenterCrop(256) pads small inputs; no resize is added','torch':torch.__version__,'cuda_build':torch.version.cuda,'compiled_cuda_architectures':torch._C._cuda_getArchFlags(),'metric_files':weights,'elapsed_seconds':time.perf_counter()-start,'scientific_result':None}
(out/'results.json').write_text(json.dumps(record,indent=2))
(out/'report.md').write_text(f'''# {out.name}: SETUP_READY

## Objective

Verify all 5,000 validation images decode and freeze the exact metric-weight bytes before GPU use.

## Setup

CPU-only job {record['job_id']} on {record['node']}; full stage {record['stage']}. Detailed identities are in results.json.

## Baseline

Official source 5c1a5693edf87228f988f363b2acdb7eb826a889; source-defined ImageNet100 validation split and released scoring helpers. No shared dataset or cache file is changed.

## Results

All 5,000 images decode; output names are unique. {small} inputs have a side smaller than 256 pixels and receive the released CenterCrop padding behavior. {len(weights)} metric files have measured SHA256 hashes. Runtime {record['elapsed_seconds']:.2f} seconds. No GPU use, model inference or paper metric result.

## Interpretation

Static input and software preparation is complete; GPU-only execution remains to be tested.

## Limitations

This audit is not a reconstruction evaluation. Small inputs are not silently resized; the custom-image demo has a different transform.

## Next action

Run one attended GPU diagnosis allocation, then release it and size final full-split batch evaluations from measured runtime and memory.
''')
print(json.dumps({k:v for k,v in record.items() if k!='metric_files'}),flush=True)
