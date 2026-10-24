"""Compare the two retained GT caches; CPU only, no model inference."""
import argparse,hashlib,json,math,os,platform,resource,time
from pathlib import Path
import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
root=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results')
folders=[root/'final-2827023-imagenet',root/'final-2827031-imagenet100']
names=[sorted(f.name for f in (d/'gt').glob('*.png')) for d in folders];assert names[0]==names[1] and len(names[0])==5000
hashes=[]
for name in names[0]:
 values=[hashlib.sha256((d/'gt'/name).read_bytes()).hexdigest() for d in folders]
 assert values[0]==values[1],name
 hashes.append({'filename':name,'sha256':values[0]})
(out/'paired-ground-truth-hashes.json').write_text(json.dumps(hashes,indent=2))
gt=[np.load(d/'ground-truth-fid-statistics.npz') for d in folders]
mu_delta=np.abs(gt[0]['mu']-gt[1]['mu']);cov_delta=np.abs(gt[0]['sigma']-gt[1]['sigma'])
conditions=['fixed_32','fixed_128','fixed_256','adaptive_0.03','adaptive_0.05','adaptive_0.09'];rows=[]
for i,d in enumerate(folders):
 original=json.loads((d/'results.json').read_text())
 for condition in conditions:
  stats=np.load(d/(condition+'-fid-statistics.npz'))
  own=float(calculate_frechet_distance(stats['mu'],stats['sigma'],gt[i]['mu'],gt[i]['sigma']))
  other=float(calculate_frechet_distance(stats['mu'],stats['sigma'],gt[1-i]['mu'],gt[1-i]['sigma']))
  assert all(math.isfinite(v) for v in [own,other])
  rows.append({'checkpoint':original['checkpoint']['dataset'],'condition':condition,'original_fid':original['reconstruction_fid'][condition],'cpu_recomputed_own_gt_fid':own,'other_run_gt_fid':other,'absolute_gt_swap_change':abs(own-other),'absolute_recompute_change':abs(own-original['reconstruction_fid'][condition])})
  print('GT cache sensitivity',rows[-1],flush=True)
result={'run_id':out.name,'node_id':'GT_CACHE_COMPARISON','status':'COMPLETED','job_id':os.getenv('SLURM_JOB_ID'),'node':platform.node(),'stage':str(Path.cwd()),'paired_ground_truth_pngs':5000,'all_paired_gt_pngs_byte_identical':True,'max_absolute_gt_feature_mean_difference':float(mu_delta.max()),'max_absolute_gt_feature_covariance_difference':float(cov_delta.max()),'cached_statistics_bitwise_equal':bool(np.array_equal(gt[0]['mu'],gt[1]['mu']) and np.array_equal(gt[0]['sigma'],gt[1]['sigma'])),'fid_sensitivity':rows,'max_observed_fid_gt_swap_change':max(v['absolute_gt_swap_change'] for v in rows),'max_observed_cpu_recompute_change':max(v['absolute_recompute_change'] for v in rows),'runtime_seconds':time.perf_counter()-start,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'interpretation':'The hash difference is evaluated using byte identity and measured cached-statistic sensitivity. These observations are not a statistical confidence interval or a bound on all numerical variation.'}
(out/'results.json').write_text(json.dumps(result,indent=2))
lines=[f'# {out.name}: GT_CACHE_COMPARISON','','## Objective','','Resolve why the two full-run ground-truth FID statistic files have different hashes.','','## Setup','',f'CPU-only job {result["job_id"]} on {result["node"]}, full stage {result["stage"]}. Read the two retained runs; no model inference, image transform or GPU command. Reuse official pytorch-fid distance on existing means/covariances.','','## Baseline','','Exactly the same frozen 5,000 ImageNet100 validation images and released PNG preprocessing were requested by both checkpoint runs. Their GT statistics were computed independently on two RTX2080Ti GPUs with source cuDNN benchmark behavior.','','## Results','',f'All 5,000 paired GT PNGs are byte-identical. Maximum absolute feature-mean difference: {result["max_absolute_gt_feature_mean_difference"]:.12g}; covariance difference: {result["max_absolute_gt_feature_covariance_difference"]:.12g}. Across both checkpoints and the six Table4 conditions, swapping the two cached GT statistics changes recomputed FID by at most {result["max_observed_fid_gt_swap_change"]:.12g}. CPU recomputation versus the recorded score changes it by at most {result["max_observed_cpu_recompute_change"]:.12g}. Full per-condition sensitivity and GT hashes are retained. Runtime {result["runtime_seconds"]:.2f} s; peak process RSS {result["host_peak_rss_kib"]} KiB.','','## Interpretation','','The statistic-file hash difference does not indicate different GT inputs. The measured sensitivity quantifies this cache difference for the six displayed comparison conditions. Small floating-point variation is consistent with independently selected source GPU kernels; its exact kernel cause was not isolated.','','## Limitations','','This is a deterministic readback/sensitivity check of existing statistics. It is not repeated inference, statistical uncertainty, a confidence interval or a universal numerical-error bound. Reconstruction FID remains separate from the primary per-image reconstruction metrics.','','## Next action','','Include the measured GT-cache sensitivity in the final reproduction evidence and close the completed released-checkpoint assignment.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='fid_sensitivity'},indent=2),flush=True)
