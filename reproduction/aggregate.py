"""Check full-run completeness and produce paper comparisons from retained arrays."""
import argparse,collections,hashlib,json,math,os,platform,resource,subprocess,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
root=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023');res=root/'karl-reproduction-results'
targets=json.loads(Path('reproduction/targets.json').read_text())
manifest_path=res/'cpu-prepare-2826204/dataset-manifest.json';manifest=json.loads(manifest_path.read_text())
expected_images={s['relative_path'] for s in manifest['samples']};assert len(expected_images)==5000
expected_names={Path(s).stem+'.png' for s in expected_images};assert len(expected_names)==5000
conditions=[x['condition'] for x in targets['conditions']];assert len(conditions)==11
metric_keys=['loglaplace','loggaussian','lpips','ssim','psnr','dreamsim']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
accounting=subprocess.check_output(['sacct','-j','2827023,2827031','--format=JobID,State,ExitCode,ElapsedRaw,MaxRSS,NodeList,AllocTRES%90','-P'],text=True)
(out/'slurm-accounting.txt').write_text(accounting)
account={r.split('|')[0]:r.split('|') for r in accounting.splitlines()[1:] if r.strip()}
runs={};checks=[];comparisons=[]
for ds,job in [('imagenet','2827023'),('imagenet100','2827031')]:
 folder=res/f'final-{job}-{ds}';x=json.loads((folder/'results.json').read_text())
 assert account[job][1:3]==['COMPLETED','0:0']
 assert x['status']=='MEASURED' and x['run_kind']=='final' and x['sample_count']==5000
 assert x['checkpoint']['dataset']==ds and x['job_id']==job
 assert set(x['metrics'])==set(conditions) and set(x['reconstruction_fid'])==set(conditions)
 assert x['fixed_halting_cutoff']==1.0 and x['adaptive_halting_cutoff']==0.75
 assert sha(x['checkpoint']['path'])==x['checkpoint']['sha256']
 stage=Path(x['stage']);metadata=json.loads((stage/'REPRODUCIBILITY_METADATA.json').read_text())
 assert metadata['git_commit_full']=='fd946c7532bd6ac90d62dcb625f38bc45e6c6ec6'
 assert {v.name for v in (folder/'gt').glob('*.png')}==expected_names
 raw=np.load(folder/'reconstruction_metrics.npz',allow_pickle=True)
 usage=json.loads((folder/'token-usage.json').read_text());assert len(usage)==55000
 groups=collections.defaultdict(list)
 for row in usage:groups[row['condition']].append(row)
 assert set(groups)==set(conditions)
 per_condition={}
 for condition in conditions:
  prefix='reconstructed_imgs_'+condition
  assert {v.name for v in (folder/prefix).glob('*.png')}==expected_names
  assert raw['total_images'].item()[prefix]==5000
  assert {Path(v).name for v in raw['all_imgs'].item()[prefix]}==expected_names
  assert len(raw['all_imgs'].item()[prefix])==5000
  for key in metric_keys:
   values=np.asarray(raw[key].item()[prefix],dtype=np.float64)
   assert values.shape==(5000,) and np.isfinite(values).all(),(ds,condition,key)
   assert math.isclose(float(values.mean()),x['metrics'][condition][key],rel_tol=0,abs_tol=1e-12),(ds,condition,key)
  rows=groups[condition];assert len(rows)==5000 and {z['image'] for z in rows}==expected_images
  tokens=np.asarray([z['active_tokens'] for z in rows]);assert ((tokens>=0)&(tokens<=[z['input_budget'] for z in rows])).all()
  assert np.equal(tokens,np.floor(tokens)).all()
  if condition.startswith('fixed_'):assert (tokens==int(condition.split('_')[1])).all()
  assert math.isclose(float(tokens.mean()),x['metrics'][condition]['active_tokens'],rel_tol=0,abs_tol=1e-12)
  assert all(math.isfinite(z['inference_seconds']) and z['inference_seconds']>0 for z in rows)
  assert math.isfinite(x['reconstruction_fid'][condition])
  stat_path=folder/(condition+'-fid-statistics.npz');stats=np.load(stat_path)
  assert stats['mu'].shape==(2048,) and stats['sigma'].shape==(2048,2048)
  assert np.isfinite(stats['mu']).all() and np.isfinite(stats['sigma']).all()
  per_condition[condition]={'reconstruction_pngs':5000,'paired_ground_truth_pngs':5000,'metric_observations_each':5000,'token_observations':5000,'active_token_min':int(tokens.min()),'active_token_max':int(tokens.max()),'fid_stats_sha256':sha(stat_path)}
 gt_stats=np.load(folder/'ground-truth-fid-statistics.npz');assert gt_stats['mu'].shape==(2048,) and gt_stats['sigma'].shape==(2048,2048)
 assert np.isfinite(gt_stats['mu']).all() and np.isfinite(gt_stats['sigma']).all()
 checks.append({'checkpoint':ds,'job_id':job,'status':'COMPLETE','conditions':per_condition,'metric_archive_sha256':sha(folder/'reconstruction_metrics.npz'),'token_usage_sha256':sha(folder/'token-usage.json'),'run_results_sha256':sha(folder/'results.json'),'checkpoint_sha256_verified':x['checkpoint']['sha256'],'gt_fid_stats_sha256':sha(folder/'ground-truth-fid-statistics.npz'),'stage_metadata':metadata})
 x['output']=str(folder);x['slurm_elapsed_seconds']=int(account[job][3]);x['slurm_host_peak_rss']=account[job+'.batch'][4];runs[ds]=x
 for i,b in enumerate(targets['table1_budgets']):
  condition='fixed_'+str(b)
  for metric,key,factor in [('l1_times_10','loglaplace',10),('lpips','lpips',1),('ssim','ssim',1),('dreamsim','dreamsim',1)]:
   actual=x['metrics'][condition][key]*factor;paper=targets['table1'][ds][metric][i]
   comparisons.append({'table':'1','checkpoint':ds,'condition':condition,'metric':metric,'actual':actual,'paper':paper,'absolute_difference':abs(actual-paper)})
 for i,e in enumerate(targets['adaptive_epsilons']):
  condition='adaptive_'+str(e)
  for metric in ['active_tokens','lpips','ssim']:
   actual=x['metrics'][condition][metric];paper=targets['table2'][ds][metric][i]
   comparisons.append({'table':'2','checkpoint':ds,'condition':condition,'metric':metric,'actual':actual,'paper':paper,'absolute_difference':abs(actual-paper)})
 for kind,values in [('fixed',targets['table1_budgets']),('adaptive',targets['adaptive_epsilons'])]:
  for i,value in enumerate(values):
   condition=kind+'_'+str(value);actual=x['reconstruction_fid'][condition];paper=targets['table4'][kind+'_rfid'][i]
   comparisons.append({'table':'4_additional_checkpoint_unspecified','checkpoint':ds,'condition':condition,'metric':'reconstruction_fid','actual':actual,'paper':paper,'absolute_difference':abs(actual-paper)})
 print('Completeness passed',ds,'5000 images x 11 conditions',flush=True)
# Read-only existing training split counts, solely for a conditional training estimate.
training_counts={}
for label,path in [('imagenet100',Path('/data/vision/torralba/datasets/imagenet100/train')),('imagenet',Path('/data/vision/torralba/datasets/imagenet_pytorch_new/train'))]:
 if path.is_dir():
  count=sum(sum(1 for f in os.scandir(c.path) if Path(f.name).suffix.lower() in {'.jpeg','.jpg','.png'}) for c in os.scandir(path) if c.is_dir())
  steps1=200*((math.ceil(count/8))//64);steps2=400*((math.ceil(count/8))//18)
  training_counts[label]={'shared_path':str(path),'count':count,'stage1_epochs':200,'stage1_per_gpu_batch':64,'stage1_loader_iterations':steps1,'stage2_epochs':400,'stage2_per_gpu_batch':18,'stage2_loader_iterations':steps2,'world_size':8,'accum_iter':1,'wall_hours_formula':f'({steps1} * measured_stage1_seconds_per_iteration + {steps2} * measured_stage2_seconds_per_iteration) / 3600','gpu_hours_formula':'8 * estimated wall hours; includes training only, checkpointing and validation extra'}
 else:training_counts[label]={'shared_path':str(path),'count':None,'reason':'Shared training path not present; checkpoint validation access does not imply training data readiness'}
# Plots: paper values are markers, released checkpoint measurements are curves.
colors={'imagenet':'#1f77b4','imagenet100':'#d55e00'};labels={'imagenet':'ImageNet1K-trained','imagenet100':'ImageNet100-trained'}
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':180})
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for ax,(title,key,factor,paper_key) in zip(axes.flat,[('L1 x 10','loglaplace',10,'l1_times_10'),('LPIPS','lpips',1,'lpips'),('SSIM','ssim',1,'ssim'),('DreamSim','dreamsim',1,'dreamsim')]):
 for ds,x in runs.items():
  ax.plot(targets['fixed_budgets'],[x['metrics']['fixed_'+str(b)][key]*factor for b in targets['fixed_budgets']],'-o',color=colors[ds],label=labels[ds]+' measured',markersize=3)
  ax.plot(targets['table1_budgets'],targets['table1'][ds][paper_key],'x',color=colors[ds],label=labels[ds]+' paper Table 1',markersize=8)
 ax.set(xlabel='Fixed active tokens',ylabel=title)
axes[0,0].legend(fontsize=8);fig.suptitle('Released KARL checkpoints: fixed-token reconstruction (5,000 ImageNet100 images)')
for suffix in ['png','pdf']:fig.savefig(out/f'fixed-token-comparison.{suffix}')
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for ax,key,title in zip(axes,['active_tokens','lpips','ssim'],['Mean active tokens','LPIPS','SSIM']):
 for ds,x in runs.items():
  ax.plot(targets['adaptive_epsilons'],[x['metrics']['adaptive_'+str(e)][key] for e in targets['adaptive_epsilons']],'-o',color=colors[ds],label=labels[ds]+' measured',markersize=4)
  ax.plot(targets['adaptive_epsilons'],targets['table2'][ds][key],'x',color=colors[ds],label=labels[ds]+' paper Table 2',markersize=8)
 ax.set(xlabel='Requested L1 epsilon',ylabel=title)
axes[0].legend(fontsize=7);fig.suptitle('Released KARL checkpoints: adaptive allocation (5,000 ImageNet100 images)')
for suffix in ['png','pdf']:fig.savefig(out/f'adaptive-comparison.{suffix}')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,kind,values,paper_values in [(axes[0],'fixed',targets['fixed_budgets'],targets['table4']['fixed_rfid']),(axes[1],'adaptive',targets['adaptive_epsilons'],targets['table4']['adaptive_rfid'])]:
 for ds,x in runs.items():ax.plot(values,[x['reconstruction_fid'][kind+'_'+str(v)] for v in values],'-o',color=colors[ds],label=labels[ds]+' measured',markersize=3)
 ax.plot(targets['table1_budgets'] if kind=='fixed' else values,paper_values,'kx',label='Paper Table 4 (checkpoint unspecified)',markersize=8)
 ax.set(xlabel='Fixed active tokens' if kind=='fixed' else 'Requested L1 epsilon',ylabel='Reconstruction FID (5,000 vs 5,000)')
axes[0].legend(fontsize=8);fig.suptitle('Additional reconstruction FID; paper checkpoint association is unspecified')
for suffix in ['png','pdf']:fig.savefig(out/f'reconstruction-fid-comparison.{suffix}')
plt.close(fig)
result={'run_id':out.name,'node_id':'COMPARISON_COMPLETE','status':'COMPLETED_WITH_MEASURED_DIFFERENCES','job_id':os.getenv('SLURM_JOB_ID'),'node':platform.node(),'stage':str(Path.cwd()),'source_revision':targets['source_revision'],'inference_source_commit':'fd946c7532bd6ac90d62dcb625f38bc45e6c6ec6','dataset_manifest_sha256':sha(manifest_path),'sample_count_each_condition':5000,'runs':runs,'completeness':checks,'paper_comparisons':comparisons,'training_estimate':training_counts,'uncertainty':targets['uncertainty'],'scientific_outcome':'Both released checkpoint evaluations are complete. Several measurements differ from the paper beyond display rounding. No arbitrary success tolerance is imposed; this is not a complete numerical reproduction.','runtime_seconds':time.perf_counter()-start,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(out/'results.json').write_text(json.dumps(result,indent=2))
lines=[f'# {out.name}: COMPARISON_COMPLETE','','## Objective','','Evaluate both primary released KARL checkpoints on the full frozen ImageNet100 validation split and compare measured reconstruction and token use with the paper.','','## Setup','',f'CPU aggregation job {result["job_id"]} on {result["node"]}; full stage {result["stage"]}. Inference jobs 2827023 and 2827031 use frozen source fd946c7532bd6ac90d62dcb625f38bc45e6c6ec6. Both are Slurm COMPLETED 0:0. Exact stages, checkpoints, hashes, runtime and raw artifacts are in results.json.','', 'Each checkpoint uses 5,000 validation images, all eleven settings, seed 0, FP32, batch 1; RGB -> CenterCrop(256) -> ToTensor(), without resize or normalization. Fixed cutoff 1.0 retains all requested tokens; adaptive cutoff .75 keeps the actual released behavior. PNG saving and metric definitions are unchanged released helpers. Full frozen protocol: reproduction/frozen-protocol.md in the staged repository.','','## Baseline','','Paper: https://arxiv.org/abs/2507.07995. Official source: https://github.com/ShivamDuggal4/karl at 5c1a5693edf87228f988f363b2acdb7eb826a889. The appropriate training-checkpoint rows of Tables 1 and 2 are primary targets. Additional reconstruction FID is compared with Table 4 without inferring its training-checkpoint association.','','## Results','','Both checkpoints have exactly 5,000 GT PNGs and 5,000 reconstruction PNGs at each of eleven conditions. Every metric/condition has 5,000 finite values and every token record is complete: 55,000 image-condition rows per checkpoint. All fixed token counts are exact, all saved aggregate means match raw observations, all FID statistics are finite, and checkpoint hashes still match the original receipts. These checks use arithmetic identity tolerances only, never paper-success thresholds.','']
for table,title in [('1','Primary Table 1: fixed budgets'),('2','Primary Table 2: adaptive allocation'),('4_additional_checkpoint_unspecified','Additional Table 4: reconstruction FID')]:
 lines += [f'### {title}','','| Checkpoint | Condition | Metric | Actual | Paper | Absolute difference |','| --- | --- | --- | ---: | ---: | ---: |']
 for row in comparisons:
  if row['table']==table:lines.append(f'| {row["checkpoint"]} | {row["condition"]} | {row["metric"]} | {row["actual"]:.6f} | {row["paper"]} | {row["absolute_difference"]:.6f} |')
 lines+=['']
lines+=['### Runtime and memory','','| Checkpoint | Slurm elapsed (s) | Inference (s) | Paired metrics (s) | FID (s) | Peak GPU allocated (bytes) | Slurm peak host RSS |','| --- | ---: | ---: | ---: | ---: | ---: | --- |']
for ds,x in runs.items():lines.append(f'| {ds} | {x["slurm_elapsed_seconds"]} | {x["generation_seconds"]:.1f} | {x["metric_seconds"]:.1f} | {x["fid_seconds"]:.1f} | {x["peak_gpu_bytes"]} | {x["slurm_host_peak_rss"]} |')
lines+=['','Standalone PNG/PDF figures: fixed-token-comparison, adaptive-comparison and reconstruction-fid-comparison. Curves show only these released checkpoints; architecture scaling or training ablations are not reproduced.','','## Interpretation','','The bounded released-checkpoint evaluation completed. Several primary metrics and token averages differ from the printed paper values; a complete numerical reproduction is not established. The ImageNet1K-trained checkpoint at 256 fixed tokens gives L1 x10 .642737 vs .68, LPIPS .148123 vs .14, SSIM .518763 vs .50 and DreamSim .129853 vs .11. The IN100-trained checkpoint at 256 tokens gives .703254/.138712/.486422/.107700 vs .70/.14/.49/.11. Adaptive token counts follow the paper trend, with exact differences retained above.','', 'Do not change checkpoints, epsilon, preprocessing or metrics to make the results closer after observing these values. These measurements support a frozen released-checkpoint baseline with explicit protocol limitations, rather than a claim that every paper number has been reproduced. No operational crash occurred in the GPU evaluations.','','## Limitations','','- The released evaluator is unfinished. The adapter repairs old names/undefined variables, uses strict inference loading, and explicitly disables fixed-budget masking. Epsilon 0 maps to .005. The paper does not specify these repairs or every evaluation detail.','- Original configured halting threshold .55 is unused by inference; adaptive execution uses the actual released .75. Standard-data preprocessing omits the resize used in the custom-image demo. There are 136 crop-padded small images.','- The paper does not pin every metric dependency or report seed variation/confidence intervals. Display rounding is precision, not statistical uncertainty or a success threshold. Source PNG truncation, direct [0,1] LPIPS and direct 256-pixel DreamSim inputs are preserved.','- Both checkpoints have833 EMA entries:440GAN-training loss entries and2obsolete unreferenced tables are excluded; all391current inference entries, including encoder/decoder/base/quantizer/halting, load strictly. No required inference weight is omitted. The installed-package DreamSim/DINO equivalence and strict official-weight proof are retained.','- Table4 does not explicitly name the training checkpoint or a separate sample count. This run uses5000reconstructions vs5000paired GT images, following the released evaluator. Both checkpoint results are shown; the closer result does not establish the missing association.','- An unmentioned fork https://github.com/kw7243/dino was created in error at2026-10-23T21:58:46Z. It and its plain local clone are preserved. Its code was never used by a model or experiment; the installed DreamSim implementation is used. No other unmentioned repository was created.','- Smoke runs remain separate. No generation evaluation, ALIT experiment, new training, experiment-code merge or PR was performed.','', 'A from-scratch run is not yet justified by these discrepancies: evaluation-setting and released-checkpoint provenance gaps remain plausible causes. If a separate training assignment becomes necessary, the released scripts specify200epochs x8GPUs x64images/GPU, followed by400epochs x8GPUs x18images/GPU, accum_iter1. The appendix reports8A100/H10080GB or4H200-class memory. Training split counts, source-loader iteration estimates and a wall/GPU-hour formula are in results.json. No honest absolute training-time estimate can be derived from inference timing; first measure representative stage1/stage2 training iterations in a separately scoped allocation.','', '## Next action','','Scope a protocol-focused follow-up around the ImageNet1K high-budget discrepancy before considering full training; retain these complete released-checkpoint results as the baseline.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'output':str(out),'status':result['status'],'runtime_seconds':result['runtime_seconds'],'training_estimate':training_counts},indent=2),flush=True)
