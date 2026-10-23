"""CPU-only strict checkpoint/model and metric-dependency checks. No forward pass."""
import argparse, json, os, platform, sys, hashlib, inspect, time
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'base_tokenizers'))
import kolmogorov_tokenizers
p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
source=Path(args.preparation)
receipts=json.loads((source/'checkpoints.json').read_text())
checks=[]
for ckpt in receipts:
    model=kolmogorov_tokenizers.karl_small(base_tokenizer_args={'id':'vqgan','is_requires_grad':False,'pretrained_ckpt_path':None},quantize_latent=True,factorize_latent=True,train_stage='evaluation')
    checkpoint=torch.load(ckpt['path'],map_location='cpu',mmap=True)
    weights=checkpoint['ema'] if 'ema' in checkpoint else checkpoint.get('model',checkpoint)
    ignored={k:v for k,v in weights.items() if k.startswith('gan_losses.')}
    inference={k:v for k,v in weights.items() if not k.startswith('gan_losses.')}
    model.load_state_dict(inference,strict=True)
    record={'dataset':ckpt['dataset'],'inference_keys':len(inference),'ignored_training_loss_keys':len(ignored),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':ckpt['path'],'checkpoint_sha256':ckpt['sha256'],'strict_inference_load':True,'rec_losses':model.rec_losses.tolist(),'configured_halting_threshold':model.dynamic_halting_threshold,'actual_halting_threshold':inspect.signature(model.prepare_decoder_latents).parameters['halting_prob'].default}
    checks.append(record);print(json.dumps(record),flush=True)
    del checkpoint,weights,inference,ignored,model
# Initialize metric weights on CPU without scoring any input.
import lpips,torchmetrics
from dreamsim import dreamsim
alex=lpips.LPIPS(net='alex').eval()
dream,dream_preprocess=dreamsim(pretrained=True,device='cpu',cache_dir=os.environ['DREAMSIM_CACHE'])
from pytorch_fid.inception import InceptionV3
fid=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
record={'node':platform.node(),'job_id':os.getenv('SLURM_JOB_ID'),'stage':str(Path.cwd()),'torch':torch.__version__,'cuda_build':torch.version.cuda,'checkpoints':checks,'lpips_source':inspect.getfile(lpips.LPIPS),'dreamsim_source':inspect.getfile(dream.__class__),'dreamsim_preprocess':str(dream_preprocess),'metric_weights_ready':True}
(out/'configuration.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
