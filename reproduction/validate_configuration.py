"""CPU-only strict checkpoint/model and metric-dependency checks. No forward pass."""
import argparse, json, os, platform, sys, hashlib, inspect, time, shutil, urllib.request
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'base_tokenizers'))
import kolmogorov_tokenizers
from reproduction.checkpoint import load_inference_state
p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
source=Path(args.preparation)
receipts=json.loads((source/'checkpoints.json').read_text())
assert len(receipts)==2
checks=[]
for ckpt in receipts:
    model=kolmogorov_tokenizers.karl_small(base_tokenizer_args={'id':'vqgan','is_requires_grad':False,'pretrained_ckpt_path':None},quantize_latent=True,factorize_latent=True,train_stage='evaluation')
    checkpoint=torch.load(ckpt['path'],map_location='cpu',mmap=True)
    weights=checkpoint['ema'] if 'ema' in checkpoint else checkpoint.get('model',checkpoint)
    ignored=load_inference_state(model,weights)
    inference={k:v for k,v in weights.items() if k not in ignored}
    record={'dataset':ckpt['dataset'],'inference_keys':len(inference),'ignored_training_only_keys':ignored,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':ckpt['path'],'checkpoint_sha256':ckpt['sha256'],'strict_inference_load':True,'rec_losses':model.rec_losses.tolist(),'configured_halting_threshold':model.dynamic_halting_threshold,'actual_halting_threshold':inspect.signature(model.prepare_decoder_latents).parameters['halting_prob'].default}
    checks.append(record)
    (out/'checkpoint-configuration.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(record),flush=True)
    del checkpoint,weights,inference,ignored,model
# Retain the existing shared cache bytes without downloading another base model.
from reproduction.metric_dependencies import DINO_WEIGHTS
existing_dino=Path('/data/scratch-fast/kwen1/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth')
DINO_WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
if not DINO_WEIGHTS.exists():shutil.copyfile(existing_dino,DINO_WEIGHTS)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
assert sha(DINO_WEIGHTS)==sha(existing_dino)
# Independently verify unknown cache ancestry against the exact public source URL,
# hashing the response stream without retaining a duplicate download.
dino_url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth'
remote_hash=hashlib.sha256()
with urllib.request.urlopen(dino_url,timeout=120) as response:
    for chunk in iter(lambda:response.read(8*1024*1024),b''):remote_hash.update(chunk)
assert remote_hash.hexdigest()==sha(DINO_WEIGHTS),'Existing DINO cache differs from official source bytes'
(out/'dino-weights.json').write_text(json.dumps({'path':str(DINO_WEIGHTS),'original_cache':str(existing_dino),'sha256':sha(DINO_WEIGHTS),'source_url':'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth','official_stream_sha256':remote_hash.hexdigest(),'provenance':'Existing user-owned Torch Hub cache, exact bytes copied to retained KARL assets and independently matched to the official source response; strict full state loading before inference'},indent=2))
# Initialize metric weights on CPU without scoring any input.
import lpips,torchmetrics
from reproduction.metric_dependencies import pin_dino_source
pin_dino_source()
from dreamsim import dreamsim
alex=lpips.LPIPS(net='alex').eval()
dream,dream_preprocess=dreamsim(pretrained=True,device='cpu',cache_dir=os.environ['DREAMSIM_CACHE'])
from pytorch_fid.inception import InceptionV3
fid=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
record={'node':platform.node(),'job_id':os.getenv('SLURM_JOB_ID'),'stage':str(Path.cwd()),'torch':torch.__version__,'cuda_build':torch.version.cuda,'checkpoints':checks,'lpips_source':inspect.getfile(lpips.LPIPS),'dreamsim_source':inspect.getfile(dream.__class__),'dreamsim_preprocess':str(dream_preprocess),'metric_weights_ready':True}
(out/'configuration.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
