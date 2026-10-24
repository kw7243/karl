import json,sys
from pathlib import Path
out=Path(sys.argv[1]);d=json.loads((out/'results.json').read_text());a=d['reconstruction']['adaptive_0.05'];b=d['reconstruction']['adaptive_0.09'];g=d['gates'];p=d['provenance'];lock=json.loads((out/'artifacts.json').read_text())
gethash=lambda suffix:next(x['sha256'] for x in lock['artifacts'] if x['path'].endswith(suffix))
ci=d['paired_bootstrap']['raw_l1']['paired_95_percentile_interval']
text=f'''## 1 Objective

Verify the KARL allocation reference in A1 of the joint adaptive tokenization plan.
This is a positive allocation reference, not a test of H_ALLOC in the new joint model.
Plan SHA-256: `71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b`.
The.09 request must use at least20% fewer tokens than.05, with counts within10% of229.6/152.5 and source LPIPS within.03 of.15/.21.
Exact checkpoint, source, data, metrics, original-slot and applicable batch coverage is required.
The coordinator owns full A1 acceptance.

## 2 Setup

- Method: KARL_IN100_SOURCE, released EMA, candidate256, batch1, requests.05/.09, actual hard halt cutoff.75.
- Run ID: `{d['run_id']}`.
- Official source: `5c1a5693edf87228f988f363b2acdb7eb826a889`; executed inference adapter: `fd946c7532bd6ac90d62dcb625f38bc45e6c6ec6`.
- Completed reference branch remains at `b1d6fa2bd3e6e5691a632d5427656fe83ed87fca`.
- New source: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-a1-20261024`; branch `research/joint-adaptive-tokenization-20261024`.
- Reduction source: `{p['source_head']}`; code SHA-256: `{gethash('/a1/reconcile.py')}`.
- Config SHA-256: `{d['config_sha256']}`; fully resolved YAML: `resolved.yaml`.
- Checkpoint SHA-256: `{d['checkpoint_sha256']}`.
- Base VQGAN: all168 embedded base state entries load strictly from this KARL EMA; no separate base checkpoint is supplied. Base config SHA-256: `{gethash('/base_tokenizers/configs/vqgan.yaml')}`; base state digest: `{lock['base_tokenizer_state_sha256']}`. Serialization is defined in base-tokenizer-state.json.
- Data:100 source-defined synsets x50 official validation images. All5000 JPEG hashes and source GT crop bytes were rechecked. Manifest SHA-256: `{d['data_manifest_sha256']}`.
- Metric protocol SHA-256: `{d['metric_protocol_sha256']}`. Source LPIPS v0.1 AlexNet receives PNG RGB tensors in[0,1], normalize=False; source SSIM data_range=1.0. The pilot AlexNet[-1,1] protocol is distinct.
- Environment: `/data/scratch-fast/kwen1/envs/karl-reproduce-20261023`; original lock SHA-256: `{gethash('/cpu-configuration-2826692/pip-freeze.txt')}`. CPU versions and pip-freeze.txt are retained.
- Native inference: FP32, no autocast, matmul TF32 off, cuDNN TF32 on, cuDNN benchmark on. Reference precision is preserved.
- Prior inference stage: `{p['prior_stage']}`. Current full repository stage: `{p['stage']}`. The complete copy includes.git and untouched official source under third_party/karl.
- Sole CPU submission/terminal and canonical KARL notebook writer: jat-a1-karl-20261024. CPU job `{p['cpu_job_id']}` on `{p['node']}`,1 CPU/4GiB/20-minute limit, no GPU. Exact argv, first/terminal script markers and logs are retained. Submission used literal --export=NIL and explicit environment after all directives.

The recorded evaluation seed remains0; the plan requests20261023.
A reachable-source proof finds no RNG draw after strict loading of all391 current inference entries and model.eval().
Explicit float requests bypass multinomial; eval bypasses random token counts; loss augmentation receives no token_add_count; VQGAN uses argmin with remap=None; dropout is off.
Seed therefore does not select a scientific input or RNG stream in these conditions.
This is a source proof, not paired-seed inference or a claim of bitwise GPU determinism.
See source-audit.md and byte-verified executed source hashes.

## 3 Baseline

Both settings use the same5000 images, checkpoint, preprocessing, metric weights, candidate count, precision and original positions; only the requested error changes.
Paper values are practical-reference anchors, not matched compute or training controls.
The fixed-count Table1 comparison does not replace this A1 test.
Source preprocessing is RGB CenterCrop(256), no resize/flip;136 small images receive zero padding.
All55000 log lines were paired to the ordered manifest, eleven condition budgets and adaptive active counts.
Only the two applicable adaptive conditions supply A1 values.

The source passes latent_tokens[:256] and the matching original positional embeddings to its decoder.
Its native per-position hard mask is unchanged: no sorting, compaction or prefix replacement occurs.
At batch1 the head-mask repeat is correct.
Larger-batch mask equivalence is unverified and is not used.
Per-image probability/mask vectors were not saved; preservation is proved by executed source, never inferred from count totals.
Exact active-slot lists and per-image nonprefix examples are unavailable.

## 4 Results

| Request | Images | Mean tokens | Source LPIPS | Source SSIM | Achieved raw L1 | Raw L1 p90 | Target violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|.05|5000|{a['mean_tokens']:.4f}|{a['source_lpips']:.6f}|{a['source_ssim']:.6f}|{a['achieved_source_raw_l1']:.6f}|{a['source_raw_l1_p90']:.6f}|{a['raw_target_violation_count']}/5000({a['raw_target_violation_rate']:.2%})|
|.09|5000|{b['mean_tokens']:.4f}|{b['source_lpips']:.6f}|{b['source_ssim']:.6f}|{b['achieved_source_raw_l1']:.6f}|{b['source_raw_l1_p90']:.6f}|{b['raw_target_violation_count']}/5000({b['raw_target_violation_rate']:.2%})|

Raw L1 is the original logged mean absolute error between[0,1] input and unclamped decoder output.
A violation is raw L1>requested value, with no fitted tolerance; FP32-request comparisons are also saved.
PNG L1 is separate because the saved images are clipped and uint8-truncated: .05={a['source_png_l1']:.6f}; .09={b['source_png_l1']:.6f}.
The source loglaplace arrays contain unscaled L1; the Table1 x10 display multiplier is not applied.
CPU readback of all10000 reconstructed PNGs agrees with their stored L1 to within{max(a['png_l1_readback_max_abs_difference'],b['png_l1_readback_max_abs_difference']):.3g}.

| Numerical gate | Measured | Required | Result |
| --- | --- | --- | --- |
|Token reduction at.09|{g['token_reduction']:.4%}|>=20%|PASS|
|Count relative errors,.05/.09|{g['relative_count_errors'][0]:.4%}/{g['relative_count_errors'][1]:.4%}|each<=10%|PASS|
|Source LPIPS absolute errors,.05/.09|{g['absolute_source_lpips_errors'][0]:.6f}/{g['absolute_source_lpips_errors'][1]:.6f}|each<=.03|PASS|

Paired raw-L1 increase(.09 minus.05):{d['paired_bootstrap']['raw_l1']['mean_difference_09_minus_05']:.6f},95% bootstrap interval[{ci[0]:.6f},{ci[1]:.6f}].
The1000 paired within-class resamples use seed917.
Other paired intervals, per-image rows, all class/count histograms and artifact hashes are retained.
These intervals describe variation over validation images, not training-seed uncertainty.

- Node status: PASS recommendation for the KARL component; full A1 acceptance is coordinator-owned.
- outcome_class:null, because no training trajectory is measured.
- New inference/training GPU-hours:0. New H100-hours:0. New inference calls:0.
- Reused whole IN100 job2827031:8464 reserved seconds={8464/3600:.6f} RTX2080Ti-hours, including all eleven conditions, source metrics and separate reconstruction FID. Per-condition GPU time is unavailable; no H100 conversion is invented.
- Prior job2827023 and aggregation2828154 were also checked terminal. IN1K results are not substituted into A1.
- New CPU analysis: runtime{d['cpu_analysis_seconds']:.3f}s, process peak RSS{d['host_peak_rss_kib']}KiB. Exact Slurm terminal accounting is in terminal-receipt.json.
- No new failed scientific run. Prior operational failures and closure evidence remain unchanged.

## 5 Interpretation

The.09 request reduces mean active tokens by{g['token_reduction']:.2%}; both counts and source LPIPS meet every declared numerical tolerance.
The original-slot, candidate256 and native-batch1 path is preserved in byte-verified source.
The configured seed difference is visible; its scientific inactivity is supported by source proof.
This supports the KARL positive reference for A1.
It does not establish a strict achieved-error guarantee, new-pilot metric equivalence, an allocation gain in joint UNITE, or a generation gain.
Violation rates are measured outcomes, not failures of an unlisted gate.
The.75 threshold acts on halting probability, not reconstruction L1.
No threshold or recipe was retuned.

## 6 Limitations

- The coordinator must accept the complete report and source proof; worker status does not accept the campaign gate.
- Actual seed0 is not a fresh20261023 run. No paired-seed GPU equality is claimed; cuDNN kernel nondeterminism is separate.
- Exact per-image active masks/probabilities are absent. A CPU reduction cannot recover them. Original-position execution is proved from source only.
- Batch>1 equivalence remains unverified; the source repeat orders multi-image masks differently from batch-major/head-minor attention. Batch1 avoids this path.
- Raw float reconstructions are absent; original full-precision scalar L1 logs and clipped/truncated PNGs are retained.
- Original ImageNet archive identity is unavailable. Exact5000 JPEG hashes, class membership and GT crops were verified; no archive hash is invented.
- Campaign recon_lpips and rfid_eval are null with reasons. Source AlexNet[0,1] LPIPS is not pilot AlexNet[-1,1]. Prior source rFID is descriptive pytorch-fid on paired5000 statistics, not campaign torch-fidelity.
- Paper settings and display values remain incompletely specified. Prior mismatches and closure are preserved; this is a practical reference gate, not exact paper reproduction.
- No optimizer, LR, generation NFE or training outcome class applies. Unavailable values are null with reasons.
- UNITE, its changed-path adapter checks and later nodes remain separately owned. This report completes only bounded KARL reconciliation.

## 7 Next action

PASS recommendation for A1 KARL: token reduction{g['token_reduction']:.4%}; count errors{g['relative_count_errors'][0]:.4%}/{g['relative_count_errors'][1]:.4%}; LPIPS errors{g['absolute_source_lpips_errors'][0]:.6f}/{g['absolute_source_lpips_errors'][1]:.6f}; native batch1 and original-slot source path verified.
Submit this report, results.json and artifacts.json to the coordinator.
No fresh inference is needed for a scientifically inactive seed.
If the coordinator requires saved per-image active masks beyond source preservation, the concrete prerequisite is a separately authorized two-condition inference pass with mask capture.
The prior whole eleven-condition job reserved2.351111 RTX2080Ti-hours, a conservative device-specific ceiling for that smaller pass on the same hardware; no H100 conversion is available.
No such job is submitted here.
Both references are required before B3/C1; this worker has no later-node assignment.

Artifacts: `{out}`. Retained files: report.md, results.json, artifacts.json, resolved.yaml, metric-protocol.json, source-audit.md, base-tokenizer-state.json, per-image-rows.json, input-image-hashes.json, terminal-receipt.json, full stage and durable Slurm log.
'''
for old,new in {'The.09':'The 0.09','by.03':'by 0.03','at least20':'at least 20','than.05':'than 0.05','within10%':'within 10%','within.03':'within 0.03','of229.6/152.5':'of 229.6/152.5','of.15/.21':'of 0.15/0.21','candidate256':'candidate 256','batch1':'batch size 1','requests.05/.09':'requests 0.05/0.09','cutoff.75':'cutoff 0.75','all168':'all 168','Data:100':'Data: 100','synsets x50':'synsets x 50','All5000':'All 5000','in[0,1]':'in [0,1]','data_range=1.0':'data_range=1.0','AlexNet[-1,1]':'AlexNet [-1,1]','all391':'all 391','remains0':'remains 0','requests20261023':'requests 20261023','same5000':'same 5000',';136':' ; 136','All55000':'All 55000','eleven condition':'eleven condition','all10000':'all 10000','source GT':'source GT','The1000':'The 1000','GPU-hours:0':'GPU-hours: 0','H100-hours:0':'H100-hours: 0','calls:0':'calls: 0','job2827031':'job 2827031','seconds8464':'seconds 8464','job2827023':'job 2827023','aggregation2828154':'aggregation 2828154','seed0':'seed 0','fresh20261023':'fresh 20261023','Exact5000':'Exact 5000','all eleven':'all eleven','paired5000':'paired 5000','reserved2.351111':'reserved 2.351111','The.75':'The 0.75','1 CPU/4GiB/20-minute':'1 CPU / 4 GiB / 20-minute','full A1':'full A1','initialization':'initialization'}.items():text=text.replace(old,new)
text=text.replace('includes.git','includes `.git`').replace('`,1 CPU','`, 1 CPU').replace('no resize/flip ;','no resize or flip;').replace('Table1','Table 1').replace('between[0,1]','between [0,1]').replace('to within{','to within {').replace('seed917','seed 917').replace('Batch1','Batch size 1').replace('native-batch size 1','native batch size 1').replace('by{','by {').replace('reduction{','reduction {').replace('errors{','errors {')
# Numbers below are formatted values; do not apply a global regex to hashes or paths.
text=text.replace('within2.85e-08','within 2.85e-08').replace('by33.61%','by 33.61%').replace('reduction33.6064%','reduction 33.6064%').replace('errors0.0260%','errors 0.0260%').replace('errors0.001994','errors 0.001994').replace('runtime814.141s, process peak RSS690852KiB','runtime 814.141 s, process peak RSS 690852 KiB').replace('job 2827031:8464','job 2827031: 8464').replace('increase(.09 minus.05):0.013320,95% bootstrap interval[0.012951,0.013730]','increase (0.09 minus 0.05): 0.013320, 95% bootstrap interval [0.012951, 0.013730]').replace('outcome_class:null','outcome_class: null').replace('0.05=','0.05 = ').replace('0.09=','0.09 = ').replace('Source AlexNet[0,1]','Source AlexNet [0,1]')
text=text.replace('Exact Slurm terminal accounting is in terminal-receipt.json.','Slurm reserved 854 s (0.237222 CPU-hours), with peak memory 4192520 KiB. The process RSS and Slurm memory measurement have different scopes. Exact terminal accounting is in terminal-receipt.json.')
(out/'report.md').write_text(text)
print(out/'report.md')
