"""A1 KARL: reduce preserved source rows; never run model inference."""
import argparse, ast, collections, hashlib, importlib.metadata, json, os, platform, re, resource, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
start=time.perf_counter();root=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023')
old=root/'karl-reproduction-results'; prior=old/'final-2827031-imagenet100'; stage=Path.cwd()
def dump(name,obj): (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def git(*argv):return subprocess.check_output(['git',*argv],text=True).strip()
assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
config=json.loads((stage/'a1/resolved.yaml').read_text());cs=sha(stage/'a1/resolved.yaml');ms=sha(stage/'a1/metric-protocol.json')
run_id='A1__KARL_IN100_SOURCE__s0__'+cs[:12]
summary=json.loads((prior/'results.json').read_text());oldstage=Path(summary['stage'])
assert summary['checkpoint']['sha256']==config['checkpoint_sha256']
assert summary['seed']==0 and summary['adaptive_halting_cutoff']==0.75
assert summary['sample_count']==5000 and summary['status']=='MEASURED'
assert sha(prior/'results.json')=='27271da1a34999bd0258a02a1f0b4d1a7267bb825bb6c3844187f26b7d47c8a5'
manifest_path=old/'cpu-prepare-2826204/dataset-manifest.json';manifest=json.loads(manifest_path.read_text());samples=manifest['samples']
ds=sha(manifest_path);assert ds=='9ced640551dc90940fdc06ba65dceff02d590ce08aaf8e5af4d34d9c25b6c2f5'
classes=collections.Counter(x['relative_path'].split('/')[0] for x in samples)
assert len(samples)==5000 and len(classes)==100 and set(classes.values())=={50}
mod=ast.parse((stage/'third_party/karl/run_scripts/create_imagenet100.py').read_text())
synsets=next(ast.literal_eval(n.value) for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='synsets' for t in n.targets))
assert set(synsets)==set(classes)==set(manifest['class_to_idx'])
assert len({Path(x['relative_path']).stem for x in samples})==5000
print('verify source and checkpoint',flush=True)
source_files={}
paths=git('ls-tree','-r','--name-only',config['source_commit']).splitlines()
for f in paths:
 data=subprocess.check_output(['git','show',config['source_commit']+':'+f]);assert (stage/'third_party/karl'/f).read_bytes()==data
 source_files[f]=hashlib.sha256(data).hexdigest()
for f in paths:
 if f not in ['notebook.md','.gitignore']:assert (oldstage/f).read_bytes()==(stage/'third_party/karl'/f).read_bytes(),f
for f in ['reproduction/evaluate_checkpoint.py','reproduction/checkpoint.py','reproduction/run_gpu.sh','reproduction/final.sbatch']:
 assert (oldstage/f).read_bytes()==(stage/f).read_bytes(),f
ckpt=Path(summary['checkpoint']['path']);assert sha(ckpt)==config['checkpoint_sha256']
state=torch.load(ckpt,map_location='cpu',mmap=True)['ema']
coverage=json.loads((stage/'reproduction/evidence/checkpoint-inference-coverage.json').read_text())
cov=next(c for c in coverage['checkpoints'] if c['dataset']=='imagenet100')
inference={k:v for k,v in state.items() if not k.startswith('gan_losses.') and k not in ['code_loss_embeddings','nll_loss_embeddings']}
assert set(inference)==set(cov['loaded_keys']) and len(inference)==391
base_hash=hashlib.sha256();base_entries=[]
for k,v in sorted(inference.items()):
 if k.startswith('base_tokenizer.'):
  header=json.dumps([k,str(v.dtype),list(v.shape)],separators=(',',':')).encode();value=v.detach().contiguous().numpy().tobytes()
  base_hash.update(header);base_hash.update(b'\0');base_hash.update(value)
  base_entries.append({'name':k,'shape':list(v.shape),'dtype':str(v.dtype),'tensor_sha256':hashlib.sha256(value).hexdigest()})
assert len(base_entries)==168
dump('base-tokenizer-state.json',{'embedded_in_checkpoint_sha256':sha(ckpt),'state_sha256':base_hash.hexdigest(),'digest_rule':'sorted name; JSON[name,dtype,shape] compact UTF8; NUL; contiguous numpy C-order bytes','entries':base_entries,'external_vqgan_checkpoint':None,'reason':'pretrained_ckpt_path=None; all base weights strictly restored from assigned KARL EMA'})
del state,inference
usage=json.loads((prior/'token-usage.json').read_text());assert len(usage)==55000
lines=(prior/'source-inference.log').read_text().splitlines();assert len(lines)==len(usage)
pattern=re.compile(r'^input tokens: (\d+) \| approx\. kolmogorov complexity: ([\d.eE+-]+) \| reconstruction l1 loss: ([\d.eE+-]+)$')
conds=['fixed_'+str(n) for n in summary['budgets']]+['adaptive_'+str(e) for e in summary['epsilons']]
assert len(conds)==11
bycond=collections.defaultdict(dict)
for i,(u,line) in enumerate(zip(usage,lines)):
 m=pattern.fullmatch(line);assert m,(i,line)
 assert u['image']==samples[i//11]['relative_path'] and u['condition']==conds[i%11]
 assert int(m[1])==u['input_budget']
 if u['condition'].startswith('adaptive_'):assert float(m[2])==u['active_tokens']
 assert u['image'] not in bycond[u['condition']]
 bycond[u['condition']][u['image']]={'raw_l1':float(m[3]),**u}
 assert np.isfinite(float(m[3]))
npz=np.load(prior/'reconstruction_metrics.npz',allow_pickle=True)
metrics={k:npz[k].item() for k in npz.files}
for c in conds:
 key='reconstructed_imgs_'+c
 assert metrics['total_images'][key]==5000
 for metric in ['loglaplace','loggaussian','lpips','ssim','psnr','dreamsim']:
  v=np.asarray(metrics[metric][key],dtype=np.float64);assert v.shape==(5000,) and np.isfinite(v).all()
  assert abs(v.mean()-summary['metrics'][c][metric])<1e-12
print('verify 5000 JPEGs, GT crop bytes and 10000 reconstruction PNGs',flush=True)
artifacts=[];hashrows=[];rows=[];condition_metrics={};arrays={};sample_by_stem={Path(s['relative_path']).stem:s for s in samples}
for i,s in enumerate(samples):
 f=Path(manifest['root'])/s['relative_path'];assert sha(f)==s['sha256']
 gt=prior/'gt'/(f.stem+'.png')
 # torchvision CenterCrop: symmetric padding then Python round-centered crop.
 im=Image.open(f).convert('RGB');w,h=im.size;padw=max(256-w,0);padh=max(256-h,0)
 if padw or padh:
  from PIL import ImageOps
  im=ImageOps.expand(im,border=(padw//2,padh//2,(padw+1)//2,(padh+1)//2),fill=0)
 w,h=im.size;left=int(round((w-256)/2.0));top=int(round((h-256)/2.0));expected=np.asarray(im.crop((left,top,left+256,top+256)),dtype=np.uint8)
 expected=(expected.astype(np.float32)/np.float32(255)*np.float32(255)).clip(0,255).astype(np.uint8)
 gt_array=np.asarray(Image.open(gt).convert('RGB'),dtype=np.uint8)
 assert np.array_equal(expected,gt_array),s['relative_path']
 hashrows.append({'image':s['relative_path'],'jpeg_sha256':s['sha256'],'gt_png_sha256':sha(gt)})
 if i%1000==0:print('data verified',i,flush=True)
for e in config['requests']:
 c='adaptive_'+str(e);folder='reconstructed_imgs_'+c;names=metrics['all_imgs'][folder]
 assert len(names)==5000 and len(set(Path(n).stem for n in names))==5000
 local=[];png_checks=[]
 for j,name in enumerate(names):
  stem=Path(name).stem;s=sample_by_stem[stem];u=bycond[c][s['relative_path']]
  recon=prior/folder/(stem+'.png');gt=prior/'gt'/(stem+'.png')
  a=np.asarray(Image.open(recon).convert('RGB'),dtype=np.int16);b=np.asarray(Image.open(gt).convert('RGB'),dtype=np.int16)
  l1=float(np.abs(a-b).mean(dtype=np.float64)/255.0);stored=float(metrics['loglaplace'][folder][j]);assert abs(l1-stored)<3e-8
  png_checks.append(abs(l1-stored))
  row={'image':s['relative_path'],'condition':c,'request':e,'active_tokens':u['active_tokens'],'raw_l1':u['raw_l1'],'raw_target_violated':u['raw_l1']>e,'png_l1':stored,'source_lpips':float(metrics['lpips'][folder][j]),'source_ssim':float(metrics['ssim'][folder][j]),'reconstruction_png_sha256':sha(recon)}
  local.append(row)
 local.sort(key=lambda x:x['image']);rows.extend(local)
 a={k:np.array([r[k] for r in local],dtype=np.float64) for k in ['active_tokens','raw_l1','png_l1','source_lpips','source_ssim']};arrays[e]=a
 hist=collections.defaultdict(collections.Counter)
 for r in local:hist[r['image'].split('/')[0]][str(r['active_tokens'])]+=1
 condition_metrics[c]={'sample_count':5000,'request':e,'mean_tokens':float(a['active_tokens'].mean()),'mean_tokens_sq':float(np.square(a['active_tokens']).mean()),'count_histogram':dict(sorted(collections.Counter(int(x) for x in a['active_tokens']).items())),'count_histogram_by_class':{k:dict(v) for k,v in hist.items()},'achieved_source_raw_l1':float(a['raw_l1'].mean()),'source_raw_l1_p90':float(np.quantile(a['raw_l1'],.9)),'raw_target_violation_count':sum(r['raw_target_violated'] for r in local),'raw_target_violation_rate':float(np.mean(a['raw_l1']>e)),'fp32_request_violation_count':int(np.sum(a['raw_l1']>float(np.float32(e)))),'source_png_l1':float(a['png_l1'].mean()),'source_png_l1_p90':float(np.quantile(a['png_l1'],.9)),'source_png_violation_rate':float(np.mean(a['png_l1']>e)),'source_lpips':float(a['source_lpips'].mean()),'source_ssim':float(a['source_ssim'].mean()),'png_l1_readback_max_abs_difference':max(png_checks),'source_rfid':summary['reconstruction_fid'][c],'source_rfid_protocol':'pytorch-fid0.3.0 paired5000; descriptive prior result; not campaign torch-fidelity0.3.0 rfid_eval'}
 print(c,{k:v for k,v in condition_metrics[c].items() if 'histogram' not in k},flush=True)
dump('per-image-rows.json',rows);dump('input-image-hashes.json',hashrows)
# Stratified paired bootstrap in sorted class/image order.
rng=np.random.RandomState(917);boot=collections.defaultdict(list)
for _ in range(1000):
 ix=(np.arange(100)[:,None]*50+rng.randint(0,50,(100,50))).reshape(-1)
 for k in arrays[.05]:boot[k].append(float((arrays[.09][k][ix]-arrays[.05][k][ix]).mean()))
boot_ci={k:{'mean_difference_09_minus_05':float((arrays[.09][k]-arrays[.05][k]).mean()),'paired_95_percentile_interval':np.quantile(v,[.025,.975]).tolist()} for k,v in boot.items()}
a=condition_metrics['adaptive_0.05'];b=condition_metrics['adaptive_0.09'];reduction=1-b['mean_tokens']/a['mean_tokens']
gates={'token_reduction':reduction,'minimum_token_reduction':.2,'reduction_pass':reduction>=.2,'relative_count_errors':[abs(a['mean_tokens']-229.6)/229.6,abs(b['mean_tokens']-152.5)/152.5],'absolute_source_lpips_errors':[abs(a['source_lpips']-.15),abs(b['source_lpips']-.21)]}
gates['numeric_pass']=gates['reduction_pass'] and max(gates['relative_count_errors'])<=.1 and max(gates['absolute_source_lpips_errors'])<=.03
assert gates['numeric_pass']
files=[manifest_path,prior/'results.json',prior/'token-usage.json',prior/'source-inference.log',prior/'reconstruction_metrics.npz',old/'aggregation-2828154/results.json',old/'cpu-configuration-2826692/pip-freeze.txt',oldstage/'REPRODUCIBILITY_METADATA.json',stage/'REPRODUCIBILITY_METADATA.json',stage/'a1/resolved.yaml',stage/'a1/metric-protocol.json',stage/'a1/reconcile.py',stage/'a1/reconcile.sbatch',stage/'a1/evidence/source-audit.md',stage/'reproduction/evidence/checkpoint-inference-coverage.json',stage/'base_tokenizers/configs/vqgan.yaml',stage/'kolmogorov_tokenizers/configs/karl_vqgan.yaml',stage/'reproduction/evaluate_checkpoint.py',stage/'reproduction/checkpoint.py',Path('/data/scratch-fast/kwen1/.cache/karl-reproduce-20261023/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'),Path('/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/lib/python3.10/site-packages/lpips/weights/v0.1/alex.pth')]
site=Path('/data/scratch-fast/kwen1/envs/karl-reproduce-20261023/lib/python3.10/site-packages')
files += [site/f for f in ['lpips/lpips.py','lpips/pretrained_networks.py','torchmetrics/image/ssim.py','torchmetrics/functional/image/ssim.py']]
for f in files:artifacts.append({'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
lock={'source_commit':config['source_commit'],'checkpoint_sha256':sha(ckpt),'data_manifest_sha256':ds,'metric_protocol_sha256':ms,'config_sha256':cs,'source_files_sha256':source_files,'artifacts':artifacts,'base_tokenizer_state_sha256':base_hash.hexdigest(),'imagenet_validation_archive_sha256':None,'imagenet_validation_archive_sha256_null_reason':'Source archive not retained in prior receipts; exact5000 JPEG content hashes are verified instead.','inception_metric_weights_sha256':None,'inception_metric_weights_sha256_null_reason':'Campaign Inception is not used in KARL count/LPIPS/SSIM/L1 reference; UNITE coordinator owns that lock.'}
dump('artifacts.json',lock)
versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','lpips','torchmetrics','Pillow','omegaconf','pytorch-fid']}
result={'node_id':'A1','scope':'KARL only; coordinator owns complete A1 acceptance','method_id':'KARL_IN100_SOURCE','run_id':run_id,'seed':0,'requested_seed':20261023,'seed_equivalence':'Source proof: no evaluation RNG draw after strict full inference-state load; seed remains0. No new paired-seed inference.','config_sha256':cs,'checkpoint_sha256':config['checkpoint_sha256'],'data_manifest_sha256':ds,'metric_protocol_sha256':ms,'node_status':'PASS','node_status_scope':'KARL component recommendation; coordinator acceptance required','outcome_class':None,'outcome_class_null_reason':'Checkpoint evaluation and CPU evidence reduction, no training trajectory','main_updates':0,'images_seen':10000,'training_gpu_hours':0.0,'gpu_hours':0.0,'reused_gpu_hours':{'device':'NVIDIA GeForce RTX2080Ti','whole_prior_job_2827031_reserved':8464/3600,'scope':'Whole eleven-condition job including source metrics and reconstruction FID; no per-condition allocation measured','h100_equivalent':None,'h100_equivalent_null_reason':'No measured conversion; no H100 charged by this task'},'training':None,'training_null_reason':'No training','generation':None,'generation_null_reason':'No KARL generation workload','reconstruction':condition_metrics,'paired_bootstrap':boot_ci,'gates':gates,'protocol_coverage':{'checkpoint':'exact assigned quantized VQGAN EMA bytes','source':'all original tracked source bytes in executed stage verified against official pin, except documentation notebook','data':'5000 source-defined official validation images;100x50; JPEG hashes and source crop GT bytes verified','batch_size':1,'larger_batch_equivalence':None,'larger_batch_equivalence_null_reason':'Not required for native batch1; source multi-image head expansion remains unverified','original_slot_identity':'Byte-verified source preserves full original indexed candidate tensor and per-slot attention mask; no sorting or compaction','per_image_slot_masks':None,'per_image_slot_masks_null_reason':'Original evaluation did not save masks/probabilities; no per-image nonprefix claim inferred from counts','seed':'scientifically inactive evaluation RNG by reachable-source proof; configured seed0 retained','source_metric':'AlexNet v0.1 LPIPS normalizeFalse on[0,1], source SSIM data_range1; separate from pilot Alex[-1,1]','raw_l1':'all55000 log lines paired to ordered manifest/conditions/counts; .05/.09 extracted','error_enforcement':'No guaranteed image-error bound; report violations independently of halt gate'},'provenance':{'prior_stage':str(oldstage),'stage':str(stage),'source_branch':'research/joint-adaptive-tokenization-20261024','source_head':git('rev-parse','HEAD'),'parent_research':'b1d6fa2bd3e6e5691a632d5427656fe83ed87fca','output':str(out),'cpu_job_id':os.environ['SLURM_JOB_ID'],'node':platform.node(),'versions':versions,'argv':sys.argv,'stage_metadata':json.loads((stage/'REPRODUCIBILITY_METADATA.json').read_text())},'cpu_analysis_seconds':time.perf_counter()-start,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'next_action':'Submit KARL evidence to coordinator; no later node or new inference assigned.'}
dump('results.json',result)
for f in ['a1/resolved.yaml','a1/metric-protocol.json','a1/evidence/source-audit.md','REPRODUCIBILITY_METADATA.json']:shutil.copy2(stage/f,out/Path(f).name)
print('A1 numerical gates',json.dumps(gates),flush=True)
print('COMPLETE',run_id,out,flush=True)
