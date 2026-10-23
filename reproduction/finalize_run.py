"""Write the required research report, including an operational failure record."""
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);code=int(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
p=out/'results.json'
result=json.loads(p.read_text()) if p.exists() else {'status':'OPERATIONAL_FAILURE','exit_code':code,'scientific_result':None,'sample_count':None,'reason':'See preserved command log; no completed scientific measurement is inferred from this crash.'}
if code:result.update(status='OPERATIONAL_FAILURE',exit_code=code)
else:result.setdefault('status','SMOKE_ONLY' if result.get('run_kind')=='smoke' else 'MEASURED')
p.write_text(json.dumps(result,indent=2))
name=out.name; targets=json.loads(Path('reproduction/targets.json').read_text())
lines=[f'# {name}: released-checkpoint-reconstruction','', '## Objective','','Measure released KARL reconstruction quality and token use. Smoke runs test execution only and cannot reproduce paper results.','','## Setup','',f'- Status: {result["status"]}.',f'- Stage: {result.get("stage",str(Path.cwd()))}.',f'- Job/node: {result.get("job_id")} / {result.get("node")}.',f'- Checkpoint: {result.get("checkpoint",{}).get("path")}; SHA256 {result.get("checkpoint",{}).get("sha256")}.',f'- Samples: {result.get("sample_count")}. Seed: 0. Batch size: 1. Float32 inference.', '- Shared ImageNet100 validation, exact source synsets; RGB, CenterCrop(256), ToTensor(), no resize/normalization.','- All metric inputs are released-style clipped, truncated uint8 PNGs.','- Each full-split per-image metric and mean active-token statistic has 5,000 observations. Each FID uses 5,000 reconstructions and their paired 5,000 ground-truth PNGs; shared ground-truth statistics are reused without changing the estimator.','- Frozen condition/table mapping: reproduction/frozen-protocol.md and reproduction/targets.json in the recorded stage.','', '## Baseline','','Official KARL source 5c1a5693edf87228f988f363b2acdb7eb826a889. Table 1 fixed budgets 32/128/256; Table 2 adaptive epsilon .03/.05/.09 at maximum 256 tokens. The driver repairs old names and undefined variables; model inference and metric helpers remain released source. Fixed-budget masking is explicitly disabled.','','## Results','']
if 'metrics' in result:
    lines += ['| Condition | L1 × 10 | LPIPS | SSIM | DreamSim | PSNR (dB) | Active tokens | Reconstruction FID |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for condition,m in result['metrics'].items():lines.append(f'| {condition} | {10*m["loglaplace"]:.6f} | {m["lpips"]:.6f} | {m["ssim"]:.6f} | {m["dreamsim"]:.6f} | {m["psnr"]:.6f} | {m["active_tokens"]:.4f} | {result.get("reconstruction_fid",{}).get(condition,"not scored")} |')
    if result.get('run_kind')=='final':
        dataset=result['checkpoint']['dataset'];lines+=['','| Metric / condition | Actual | Paper | Absolute difference |','| --- | ---: | ---: | ---: |']
        for t in targets['table1_budgets']:
            for metric,key,factor in [('l1_times_10','loglaplace',10),('lpips','lpips',1),('ssim','ssim',1),('dreamsim','dreamsim',1)]:
                actual=result['metrics']['fixed_'+str(t)][key]*factor;paper=targets['table1'][dataset][metric][targets['table1_budgets'].index(t)]
                lines.append(f'| {metric} / {t} tokens | {actual:.6f} | {paper:.2f} | {abs(actual-paper):.6f} |')
        for i,e in enumerate(targets['adaptive_epsilons']):
            for metric in ['active_tokens','lpips','ssim']:
                actual=result['metrics']['adaptive_'+str(e)][metric];paper=targets['table2'][dataset][metric][i]
                lines.append(f'| {metric} / epsilon {e} | {actual:.6f} | {paper} | {abs(actual-paper):.6f} |')
    if result.get('run_kind')=='final':
        lines+=['','Additional Table 4 reconstruction-FID comparison (training-checkpoint association unspecified in paper):','','| Condition | Actual reconstruction FID | Paper | Absolute difference |','| --- | ---: | ---: | ---: |']
        for i,t in enumerate(targets['table1_budgets']):
            condition='fixed_'+str(t);actual=result['reconstruction_fid'][condition];paper=targets['table4']['fixed_rfid'][i]
            lines.append(f'| {condition} | {actual:.6f} | {paper} | {abs(actual-paper):.6f} |')
        for i,e in enumerate(targets['adaptive_epsilons']):
            condition='adaptive_'+str(e);actual=result['reconstruction_fid'][condition];paper=targets['table4']['adaptive_rfid'][i]
            lines.append(f'| {condition} | {actual:.6f} | {paper} | {abs(actual-paper):.6f} |')
    lines += ['',f'- Runtime: {result.get("total_seconds")} s; generation {result.get("generation_seconds")} s; paired metrics {result.get("metric_seconds")} s; FID {result.get("fid_seconds")} s.',f'- Peak GPU allocated: {result.get("peak_gpu_bytes")} bytes. Host peak RSS: {result.get("host_peak_rss_kib")} KiB.']
else:lines += [result.get('reason','The command did not complete; see the preserved log.')]
lines += ['','## Interpretation','', 'Execution check only; no paper result.' if result.get('run_kind')=='smoke' else ('No scientific conclusion follows from an operational crash.' if code else 'Measured reconstruction and token use are compared directly with the paper. Numerical differences are evidence; no arbitrary reproduction threshold is applied.'),'','## Limitations','','- Paper does not pin all metric dependency versions or report repeated-seed uncertainty. Display rounding is not statistical uncertainty.','- Fixed-count paper behavior requires an explicit adapter because released epsilon=0 retains halting. Adaptive cutoff uses released .75, rather than the unused YAML value.','- Released data preprocessing omits the resize present in the custom-image demo.','- Reconstruction FID is distinct from generation FID. Table 4 does not explicitly label its training-checkpoint association.','- No ALIT experiment, new training, source-code merge, or PR is part of this run.','','## Next action','', 'Resolve the logged operational failure before another experimental attempt.' if code else ('Use measured runtime and memory to size the full evaluation after all smoke checks pass.' if result.get('run_kind')=='smoke' else 'Compare the second released VQGAN checkpoint and consolidate the project report.')]
(out/'report.md').write_text('\n'.join(lines)+'\n')
