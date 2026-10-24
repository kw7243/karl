## 1 Objective

Verify the KARL allocation reference in A1 of the joint adaptive tokenization plan.
This is a positive allocation reference, not a test of H_ALLOC in the new joint model.
Plan SHA-256: `71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b`.
The 0.09 request must use at least 20% fewer tokens than 0.05, with counts within 10% of 229.6/152.5 and source LPIPS within 0.03 of 0.15/0.21.
Exact checkpoint, source, data, metrics, original-slot and applicable batch coverage is required.
The coordinator owns full A1 acceptance.

## 2 Setup

- Method: KARL_IN100_SOURCE, released EMA, candidate 256, batch size 1, requests 0.05/0.09, actual hard halt cutoff 0.75.
- Run ID: `A1__KARL_IN100_SOURCE__s0__e62fe7f422b6`.
- Official source: `5c1a5693edf87228f988f363b2acdb7eb826a889`; executed inference adapter: `fd946c7532bd6ac90d62dcb625f38bc45e6c6ec6`.
- Completed reference branch remains at `b1d6fa2bd3e6e5691a632d5427656fe83ed87fca`.
- New source: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-a1-20261024`; branch `research/joint-adaptive-tokenization-20261024`.
- Reduction source: `9b6533d5f47751772db99c40cf85aad1be485b7c`; code SHA-256: `3c512b92dad856ab96c911c590d5f17c6edd9722529decafe63e68b4d9b520fa`.
- Config SHA-256: `e62fe7f422b62bf931f71a2dcf1db61831c2058d566273d896556e5b1c47f22f`; fully resolved YAML: `resolved.yaml`.
- Checkpoint SHA-256: `a34b4b9ba1d61c5e58aecb96fd253bdc6d9c2e9156a6f13c42a8ae8f472512ae`.
- Base VQGAN: all 168 embedded base state entries load strictly from this KARL EMA; no separate base checkpoint is supplied. Base config SHA-256: `3a544b7f4335bc33ad9f184607ecb779fbc153a9217472cea27c7a12565dcf24`; base state digest: `904ad13e62f406edea2aeac152c0b5ef965f5584fc224b9895b1487cfbc7cd14`. Serialization is defined in base-tokenizer-state.json.
- Data: 100 source-defined synsets x 50 official validation images. All 5000 JPEG hashes and source GT crop bytes were rechecked. Manifest SHA-256: `9ced640551dc90940fdc06ba65dceff02d590ce08aaf8e5af4d34d9c25b6c2f5`.
- Metric protocol SHA-256: `8372b57a5dcfe55c650647e630c7abe8420a0bcf5e8536d30360003ea6e7c66f`. Source LPIPS v0.1 AlexNet receives PNG RGB tensors in [0,1], normalize=False; source SSIM data_range=1.0. The pilot AlexNet [-1,1] protocol is distinct.
- Environment: `/data/scratch-fast/kwen1/envs/karl-reproduce-20261023`; original lock SHA-256: `c8ca1bddd93a712e78ffce278f4e13b199d60693683f02e0f481715939067f49`. CPU versions and pip-freeze.txt are retained.
- Native inference: FP32, no autocast, matmul TF32 off, cuDNN TF32 on, cuDNN benchmark on. Reference precision is preserved.
- Prior inference stage: `/data/scratch-fast/kwen1/karl-reproduce-20261023-stages/20261023_224128-cb7936-fd946c7-code`. Current full repository stage: `/data/scratch-fast/kwen1/karl-a1-20261024-stages/20261024_061901-0e2f89-9b6533d-code`. The complete copy includes `.git` and untouched official source under third_party/karl.
- Sole CPU submission/terminal and canonical KARL notebook writer: jat-a1-karl-20261024. CPU job `2830332` on `groenig-3.csail.mit.edu`, 1 CPU / 4 GiB / 20-minute limit, no GPU. Exact argv, first/terminal script markers and logs are retained. Submission used literal --export=NIL and explicit environment after all directives.

The recorded evaluation seed remains 0; the plan requests 20261023.
A reachable-source proof finds no RNG draw after strict loading of all 391 current inference entries and model.eval().
Explicit float requests bypass multinomial; eval bypasses random token counts; loss augmentation receives no token_add_count; VQGAN uses argmin with remap=None; dropout is off.
Seed therefore does not select a scientific input or RNG stream in these conditions.
This is a source proof, not paired-seed inference or a claim of bitwise GPU determinism.
See source-audit.md and byte-verified executed source hashes.

## 3 Baseline

Both settings use the same 5000 images, checkpoint, preprocessing, metric weights, candidate count, precision and original positions; only the requested error changes.
Paper values are practical-reference anchors, not matched compute or training controls.
The fixed-count Table 1 comparison does not replace this A1 test.
Source preprocessing is RGB CenterCrop(256), no resize or flip; 136 small images receive zero padding.
All 55000 log lines were paired to the ordered manifest, eleven condition budgets and adaptive active counts.
Only the two applicable adaptive conditions supply A1 values.

The source passes latent_tokens[:256] and the matching original positional embeddings to its decoder.
Its native per-position hard mask is unchanged: no sorting, compaction or prefix replacement occurs.
At batch size 1 the head-mask repeat is correct.
Larger-batch mask equivalence is unverified and is not used.
Per-image probability/mask vectors were not saved; preservation is proved by executed source, never inferred from count totals.
Exact active-slot lists and per-image nonprefix examples are unavailable.

## 4 Results

| Request | Images | Mean tokens | Source LPIPS | Source SSIM | Achieved raw L1 | Raw L1 p90 | Target violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|.05|5000|229.6596|0.151994|0.473560|0.073647|0.113508|3819/5000(76.38%)|
|.09|5000|152.4792|0.206149|0.422696|0.086967|0.113972|1611/5000(32.22%)|

Raw L1 is the original logged mean absolute error between [0,1] input and unclamped decoder output.
A violation is raw L1>requested value, with no fitted tolerance; FP32-request comparisons are also saved.
PNG L1 is separate because the saved images are clipped and uint8-truncated: .05=0.073574; .09=0.086888.
The source loglaplace arrays contain unscaled L1; the Table 1 x10 display multiplier is not applied.
CPU readback of all 10000 reconstructed PNGs agrees with their stored L1 to within 2.85e-08.

| Numerical gate | Measured | Required | Result |
| --- | --- | --- | --- |
|Token reduction at.09|33.6064%|>=20%|PASS|
|Count relative errors,.05/.09|0.0260%/0.0136%|each<=10%|PASS|
|Source LPIPS absolute errors,.05/.09|0.001994/0.003851|each<=.03|PASS|

Paired raw-L1 increase (0.09 minus 0.05): 0.013320, 95% bootstrap interval [0.012951, 0.013730].
The 1000 paired within-class resamples use seed 917.
Other paired intervals, per-image rows, all class/count histograms and artifact hashes are retained.
These intervals describe variation over validation images, not training-seed uncertainty.

- Node status: PASS recommendation for the KARL component; full A1 acceptance is coordinator-owned.
- outcome_class: null, because no training trajectory is measured.
- New inference/training GPU-hours: 0. New H100-hours: 0. New inference calls: 0.
- Reused whole IN100 job 2827031: 8464 reserved seconds=2.351111 RTX2080Ti-hours, including all eleven conditions, source metrics and separate reconstruction FID. Per-condition GPU time is unavailable; no H100 conversion is invented.
- Prior job 2827023 and aggregation 2828154 were also checked terminal. IN1K results are not substituted into A1.
- New CPU analysis: runtime 814.141 s, process peak RSS 690852 KiB. Slurm reserved 854 s (0.237222 CPU-hours), with peak memory 4192520 KiB. The process RSS and Slurm memory measurement have different scopes. Exact terminal accounting is in terminal-receipt.json.
- No new failed scientific run. Prior operational failures and closure evidence remain unchanged.

## 5 Interpretation

The 0.09 request reduces mean active tokens by 33.61%; both counts and source LPIPS meet every declared numerical tolerance.
The original-slot, candidate 256 and native batch size 1 path is preserved in byte-verified source.
The configured seed difference is visible; its scientific inactivity is supported by source proof.
This supports the KARL positive reference for A1.
It does not establish a strict achieved-error guarantee, new-pilot metric equivalence, an allocation gain in joint UNITE, or a generation gain.
Violation rates are measured outcomes, not failures of an unlisted gate.
The 0.75 threshold acts on halting probability, not reconstruction L1.
No threshold or recipe was retuned.

## 6 Limitations

- The coordinator must accept the complete report and source proof; worker status does not accept the campaign gate.
- Actual seed 0 is not a fresh 20261023 run. No paired-seed GPU equality is claimed; cuDNN kernel nondeterminism is separate.
- Exact per-image active masks/probabilities are absent. A CPU reduction cannot recover them. Original-position execution is proved from source only.
- Batch>1 equivalence remains unverified; the source repeat orders multi-image masks differently from batch-major/head-minor attention. Batch size 1 avoids this path.
- Raw float reconstructions are absent; original full-precision scalar L1 logs and clipped/truncated PNGs are retained.
- Original ImageNet archive identity is unavailable. Exact 5000 JPEG hashes, class membership and GT crops were verified; no archive hash is invented.
- Campaign recon_lpips and rfid_eval are null with reasons. Source AlexNet [0,1] LPIPS is not pilot AlexNet [-1,1]. Prior source rFID is descriptive pytorch-fid on paired 5000 statistics, not campaign torch-fidelity.
- Paper settings and display values remain incompletely specified. Prior mismatches and closure are preserved; this is a practical reference gate, not exact paper reproduction.
- No optimizer, LR, generation NFE or training outcome class applies. Unavailable values are null with reasons.
- UNITE, its changed-path adapter checks and later nodes remain separately owned. This report completes only bounded KARL reconciliation.

## 7 Next action

PASS recommendation for A1 KARL: token reduction 33.6064%; count errors 0.0260%/0.0136%; LPIPS errors 0.001994/0.003851; native batch size 1 and original-slot source path verified.
Submit this report, results.json and artifacts.json to the coordinator.
No fresh inference is needed for a scientifically inactive seed.
If the coordinator requires saved per-image active masks beyond source preservation, the concrete prerequisite is a separately authorized two-condition inference pass with mask capture.
The prior whole eleven-condition job reserved 2.351111 RTX2080Ti-hours, a conservative device-specific ceiling for that smaller pass on the same hardware; no H100 conversion is available.
No such job is submitted here.
Both references are required before B3/C1; this worker has no later-node assignment.

Artifacts: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-a1-20261024-results/A1/cpu-2830332`. Retained files: report.md, results.json, artifacts.json, resolved.yaml, metric-protocol.json, source-audit.md, base-tokenizer-state.json, per-image-rows.json, input-image-hashes.json, terminal-receipt.json, full stage and durable Slurm log.
