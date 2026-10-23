"""Paper reconstruction protocol using released KARL inference and scoring functions.

Repairs only the unfinished evaluation driver. Fixed-token runs suppress halting
explicitly; adaptive runs retain the released default probability cutoff (0.75).
No model, learned weight, reconstruction rule, or metric formula is replaced.
"""
import argparse, contextlib, functools, hashlib, json, os, platform, re, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'base_tokenizers'))
import kolmogorov_tokenizers
# The released evaluator retained the old import name; use its metric/save helpers.
sys.modules['adaptive_tokenizers']=kolmogorov_tokenizers
import evaluate as released
from reproduction.metric_dependencies import pin_dino_source
pin_dino_source()
from dreamsim import dreamsim
released.dreamsim=functools.partial(dreamsim,cache_dir=os.environ['DREAMSIM_CACHE'])

p=argparse.ArgumentParser()
p.add_argument('--checkpoint-manifest',required=True)
p.add_argument('--dataset-manifest',required=True)
p.add_argument('--checkpoint-dataset',choices=['imagenet','imagenet100'],required=True)
p.add_argument('--output',required=True)
p.add_argument('--mode',choices=['smoke','final'],required=True)
p.add_argument('--limit',type=int)
p.add_argument('--budgets',default='32,64,96,128,160,192,224,256')
p.add_argument('--epsilons',default='0.03,0.05,0.09')
args=p.parse_args()
if args.mode=='final': assert args.limit is None,'A paper evaluation must use the full split'
else: assert args.limit and args.limit<=32,'Smoke requires an explicit small count'
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
manifest=json.loads(Path(args.dataset_manifest).read_text()); samples=manifest['samples']
assert len(samples)==5000
if args.limit:samples=samples[:args.limit]
checkpoint=next(x for x in json.loads(Path(args.checkpoint_manifest).read_text()) if x['dataset']==args.checkpoint_dataset)
assert checkpoint['sha256']
torch.manual_seed(0);np.random.seed(0)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True
assert torch.cuda.is_available()
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter()
model=kolmogorov_tokenizers.karl_small(base_tokenizer_args={'id':'vqgan','is_requires_grad':False,'pretrained_ckpt_path':None},quantize_latent=True,factorize_latent=True,train_stage='evaluation')
ckpt=torch.load(checkpoint['path'],map_location='cpu',mmap=True)
weights=ckpt['ema']
model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('gan_losses.')},strict=True)
del ckpt,weights
model=model.eval().cuda()
original_prepare=model.prepare_decoder_latents
latest={}
def prepare(*values,**kwargs):
    kwargs['halting_prob']=1.0 if latest['fixed'] else 0.75
    result=original_prepare(*values,**kwargs)
    latest['active']=(result[1][...,0]<=kwargs['halting_prob']).sum(dim=-1).detach().cpu().tolist()
    return result
model.prepare_decoder_latents=prepare
# This is precisely the released standard-dataset path, including padding for
# images smaller than 256 pixels. The custom-image resize path is not used.
transform=transforms.Compose([transforms.CenterCrop(256),transforms.ToTensor()])
budgets=[int(x) for x in args.budgets.split(',')]
epsilons=[float(x) for x in args.epsilons.split(',')]
conditions=[('fixed_'+str(t),t,0.0,True) for t in budgets]+[('adaptive_'+str(e),256,e,False) for e in epsilons]
for name,_,_,_ in conditions:(out/('reconstructed_imgs_'+name)).mkdir()
(out/'gt').mkdir()
usage=[]
with (out/'source-inference.log').open('w') as log,torch.no_grad():
    for idx,sample in enumerate(samples):
        path=Path(manifest['root'])/sample['relative_path']
        image=transform(Image.open(path).convert('RGB')).unsqueeze(0).cuda()
        filename=path.stem+'.png'
        released.save_imgs(image,str(out/'gt'/filename))
        for name,t,epsilon,fixed in conditions:
            latest['fixed']=fixed
            torch.cuda.synchronize(); then=time.perf_counter()
            with contextlib.redirect_stdout(log):_,reconstructions,logs=model.encode(image,input_token_budget=[t],desired_reconstruction_quality=epsilon)
            torch.cuda.synchronize();elapsed=time.perf_counter()-then
            assert len(reconstructions)==1
            recon=reconstructions[0]
            assert tuple(recon.shape)==(1,3,256,256) and bool(torch.isfinite(recon).all())
            if fixed:assert latest['active']==[t]
            released.save_imgs(recon,str(out/('reconstructed_imgs_'+name)/filename))
            usage.append({'image':sample['relative_path'],'condition':name,'input_budget':t,'epsilon':epsilon,'active_tokens':latest['active'][0],'inference_seconds':elapsed})
        if idx%100==0:print('generation',idx+1,len(samples),flush=True)
(out/'token-usage.json').write_text(json.dumps(usage))
generation_seconds=time.perf_counter()-start
peak_generation=torch.cuda.max_memory_allocated()
del model,image,recon,reconstructions,logs
import gc;gc.collect();torch.cuda.empty_cache()
# Reuse released PNG-based metric definitions. LPIPS receives [0,1] with its
# default normalize=False; DreamSim receives the same PNG tensors directly.
metric_start=time.perf_counter()
with torch.no_grad():
    released.compute_reconstruction_losses(str(out))
metric_seconds=time.perf_counter()-metric_start
fid={}
if args.mode=='final':
    from pytorch_fid.fid_score import calculate_fid_given_paths
    for name,_,_,_ in conditions:
        fid[name]=float(calculate_fid_given_paths([str(out/('reconstructed_imgs_'+name)),str(out/'gt')],batch_size=50,device=torch.device('cuda'),dims=2048,num_workers=4))
summary={'run_kind':args.mode,'checkpoint':checkpoint,'sample_count':len(samples),'seed':0,'stage':str(Path.cwd()),'job_id':os.getenv('SLURM_JOB_ID'),'node':platform.node(),'gpu':torch.cuda.get_device_name(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda_build':torch.version.cuda,'preprocessing':'RGB; CenterCrop(256); ToTensor; no resize/normalization','png_encoding':'clip([0,1])*255 then uint8 truncation, released save_imgs','fixed_halting_cutoff':1.0,'adaptive_halting_cutoff':0.75,'budgets':budgets,'epsilons':epsilons,'generation_seconds':generation_seconds,'metric_seconds':metric_seconds,'total_seconds':time.perf_counter()-start,'peak_generation_gpu_bytes':peak_generation,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'reconstruction_fid':fid}
metrics=np.load(out/'reconstruction_metrics.npz',allow_pickle=True)
summary['metrics']={}
for name,_,_,_ in conditions:
    folder='reconstructed_imgs_'+name
    summary['metrics'][name]={key:float(np.mean(metrics[key].item()[folder])) for key in ['loglaplace','loggaussian','lpips','ssim','psnr','dreamsim']}
    summary['metrics'][name]['active_tokens']=float(np.mean([x['active_tokens'] for x in usage if x['condition']==name]))
(out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
