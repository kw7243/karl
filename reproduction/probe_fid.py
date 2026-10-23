"""SMOKE ONLY: verify the released FID default batch fits, no FID measurement."""
import json,os,platform,time
from pathlib import Path
import torch
from pytorch_fid.inception import InceptionV3
out=Path(os.environ['PROBE_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
torch.hub.set_dir(os.environ['DREAMSIM_CACHE'])
torch.manual_seed(0)
assert torch.cuda.is_available()
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter()
model=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().cuda()
with torch.no_grad():
 x=torch.zeros(50,3,256,256,device='cuda')
 y=model(x)[0]
 assert list(y.shape)==[50,2048,1,1] and bool(torch.isfinite(y).all())
 torch.cuda.synchronize()
result={'run_kind':'smoke','status':'SMOKE_ONLY','job_id':os.environ['SLURM_JOB_ID'],'node':platform.node(),'stage':str(Path.cwd()),'gpu':torch.cuda.get_device_name(),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'batch':50,'input_shape':list(x.shape),'output_shape':list(y.shape),'finite':True,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-start,'scientific_result':None}
(out/'results.json').write_text(json.dumps(result,indent=2))
(out/'report.md').write_text(f"# {out.name}: FID-batch-memory-smoke\n\n## Objective\n\nVerify the released default FID batch of 50 fits GPU memory.\n\n## Setup\n\nJob {result['job_id']}, node {result['node']}, stage {result['stage']}; pretrained official pytorch-fid Inception, FP32.\n\n## Baseline\n\nReleased compute_reconstruction_fid invokes pytorch-fid default batch 50 and 2048 features.\n\n## Results\n\nFinite features with shape {result['output_shape']}; peak GPU allocation {result['peak_gpu_bytes']} bytes; elapsed {result['elapsed_seconds']:.2f} s.\n\n## Interpretation\n\nThe default scoring batch fits. No FID or paper measurement was computed.\n\n## Limitations\n\nSynthetic zeros are an execution check only. Full sample statistics and CPU matrix operations remain unmeasured.\n\n## Next action\n\nSize and submit the full released-checkpoint evaluations after both checkpoint smoke checks pass.\n")
print(json.dumps(result,indent=2),flush=True)
