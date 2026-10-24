# KARL research notebook

## Current state

- Released-checkpoint assignment completed on 2026-10-24 UTC; both full GPU jobs and CPU validation jobs are terminal.
- Complete numerical agreement with the paper is not established. Exact actual/paper/difference tables and figures: /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/aggregation-2828154/.
- All raw data, checkpoints, environments, complete stages and results remain on CSAIL. No active job remains owned by this assignment after the completion handoff.
- Next action: resolve the documented IN1K high-budget protocol discrepancy before considering full training.

## Source setup receipt: 2026-10-23

- Date: 2026-10-23 UTC.
- Role: Source prepared for a separate reproduction worker.
- Official upstream: https://github.com/ShivamDuggal4/karl.
- Public fork: https://github.com/kw7243/karl.
- Pinned upstream source: `5c1a5693edf87228f988f363b2acdb7eb826a889`.
- Canonical retained clone: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl`.
- SSH access: `kwen1@slurm-login.csail.mit.edu` (observed `slurm-login-0.csail.mit.edu`).
- The authenticated GitHub account is `kw7243`. GitHub API confirms the fork parent is `ShivamDuggal4/karl`.
- This notebook is the only setup change. No environment, checkpoint download, experiment, or GPU allocation was started.

## Branch index

- `main`: canonical source plus this documentation-only setup commit; fork and upstream default branches remain `main`.
- `upstream/main`: official source at setup revision `5c1a5693edf87228f988f363b2acdb7eb826a889`.
- `research/karl-reproduce-20261023`: independent plain clone, sole experiment branch; ancestry begins at notebook-only main `9f6e199c7f85db1f6509f6a92c417c00ff81c0a2`. All experiment code and reports are indexed in `reproduction/evidence/run-index.json`; main receives notebook changes only.
- No experiment code was merged. No force push was used.

## Storage and next step

- Retained source stays under `/data/vision/torralba/u/kwen1/image-tokenization-20261023`.
- Performance copies, environments, and caches belong under `/data/scratch-fast/kwen1`.
- Stage a complete repository copy before any later experiment, as required by the research reproducibility skill.
- The next reproduction worker must inspect the pinned README, configuration, training, and evaluation source before selecting the scientific protocol. No experiments have run.
- Start with `README.md` at the pinned source. Checkpoint URLs and evaluation notes are source references, not verified downloads.
- CSAIL GitHub SSH authentication was unavailable during setup. This notebook was committed through the authenticated GitHub API and then fetched into the retained clone. No credential was copied to CSAIL.

## 2026-10-23: released-checkpoint reproduction intake

- Sole experiment submitter, terminal owner and canonical notebook writer: `karl-reproduce-20261023`.
- Branch `research/karl-reproduce-20261023` descends from notebook-only main `9f6e199c7f85db1f6509f6a92c417c00ff81c0a2`, whose parent is official source `5c1a5693edf87228f988f363b2acdb7eb826a889`.
- Independent plain clone: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduce-20261023`; no pooled worktree operation.
- Existing CSAIL HTTPS credential helper passed a branch-backup dry run. GitHub SSH is unavailable; no new credential is needed.
- Initial Table 1 target: released ImageNet1K-trained quantized VQGAN KARL evaluated on ImageNet100, 5,000 validation images, budgets 32/128/256. Paper L1×10 = 1.10/0.81/0.68; LPIPS = .29/.18/.14; SSIM = .35/.44/.50; DreamSim = .27/.14/.11.
- Source `run_scripts/create_imagenet100.py` defines exactly the 100 classes already available read-only at `/data/vision/torralba/datasets/imagenet100/val`; 50 images per class. No new shared dataset is needed.
- Five released checkpoint URLs return binary assets. Download and inspect the two primary quantized VQGAN checkpoints first; do not claim a checkpoint hash until measured.
- `evaluate.py` has stale `adaptive_tokenizers`/`alit_small`, undefined `evaluation_logs`/`image_active_token_count`, and calls a two-argument save function with three arguments. Its standard-data transform is CenterCrop(256), ToTensor() with no resize or normalization. Demo transform additionally resizes. These are unresolved evaluation fidelity differences.
- Model inference uses a hardcoded .75 halting probability; configured .55 is not consumed. Fixed-budget mode is not explicitly disabled by epsilon=0 in the model. Resolve this against checkpoint configuration before GPU use.
- CPU preparation uses a full stage, isolated environment and retained checkpoints/results. No reconstruction, smoke result or GPU allocation has run.
- Paper has no reported seed variance or confidence intervals. Display rounding alone is ±.005 for two-decimal displayed metrics; this is a precision interval, not a success threshold. Report measured differences without inventing tolerance.

### CPU preparation receipt

- Job `2826204` was accepted and started on `groenig-2`, CPU only, 4 CPUs, 16 GiB, 1 hour maximum, partition `tig-cpu`, account `csail`, QoS `tig-main`.
- Submission/terminal owner: `karl-reproduce-20261023`; no other worker may resubmit or declare its terminal result.
- Stage `/data/scratch-fast/kwen1/karl-reproduce-20261023-stages/20261023_214235-78593e-736a2af-code`, source `736a2af`.
- Output `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-prepare-2826204/`; Slurm log is adjacent as `cpu-prepare-2826204.slurm.log`.
- Earlier stage `20261023_174203-16d9b8-70ca972-code` stopped before submission because archive-mode rsync could not restore retained-directory setgid permissions on scratch. It is preserved. Reused the existing storage compatibility fix: project-local copy of the installed staging helper changes only rsync permission/ownership restoration, preserving all contents, symlinks, timestamps and Git metadata. Evidence: `reproduction/evidence/staging-compatibility.json`.
- No GPU request, reconstruction, scoring, or paper result exists yet.

### Frozen evaluation driver and metric source preparation

- `reproduction/targets.json` records both primary quantized VQGAN checkpoints, eight released evaluator budgets (32 through 256 in steps of 32), Table 1 targets at 32/128/256, and Table 2 targets at epsilon .03/.05/.09 with maximum 256 tokens.
- Fixed counts explicitly use halting cutoff 1.0; adaptive inference retains the released hardcoded .75. Requested epsilon 0 is discretized by the released model to .005. No model weights or inference algorithm are replaced.
- Reuse released PNG saving and reconstruction metric functions. Metric calls run without gradient recording. Keep per-image measurements, token use, PNG encoding, runtime and peak memory; smoke outputs cannot be paper results.
- Full CPU stage provenance verified 55 tracked file blobs against Git and retained meaningful executable bits. Receipt: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-stage-provenance.json`.
- DreamSim 0.2.1 ordinarily fetches DINO from a mutable Torch Hub source. Its installed package already bundles the required implementation. All nine relevant class/function syntax trees match the reference source; initialization alone differs and must be completely replaced by strict pretrained-state loading before inference. Use the existing user-owned cached base weights; freeze and verify their identity during CPU configuration. Evidence: `reproduction/evidence/dino-package-equivalence.json`.
- The unused DINO source copy is not an experiment dependency. The evaluation path denies unplanned dynamic Torch Hub source loads.
- CPU preparation job 2826204 remains active; no GPU or paper evaluation has run.

### CPU preparation completed; strict configuration in progress

- CPU job 2826204 completed with Slurm exit 0:0 in 1,861 seconds, peak host RSS 9,993,332 KiB, zero GPU use. Its read-only process-inspection step had ps reject the Slurm extern PID sentinel -1; no preparation command failed.
- Compute-node software: Python 3.10.22, Torch 2.3.0+cu121, CUDA build 12.1, NumPy 1.24.1; full initial environment lock is retained with the run. The source's older pip 20.3 was replaced by pip 23.3.2 for legacy package metadata support.
- ImageNet1K-trained quantized VQGAN checkpoint: 1,643,547,191 bytes, SHA256 `c5ff98deb0e40c3307e04c5ff5a0f54cccac40e09ada83d19bbc3e2df0ae1600`.
- ImageNet100-trained quantized VQGAN checkpoint: 1,643,547,191 bytes, SHA256 `a34b4b9ba1d61c5e58aecb96fd253bdc6d9c2e9156a6f13c42a8ae8f472512ae`.
- Both releases contain EMA only: 833 keys / 410,799,328 tensor elements including 440 GAN-loss keys. No epoch or training arguments are saved. The source model uses 1024-wide positional embeddings and 4096 × 12 latent codebook entries, consistent with inspected checkpoint shapes.
- Two old conditioning tables, `code_loss_embeddings` and `nll_loss_embeddings`, are absent from current inference code. Exclude only these exact keys and training GAN-loss modules; strict loading must still accept every current inference parameter and reject every other mismatch. Source evaluator instead uses unrestricted strict=False.
- Dataset manifest: all 5,000 images hashed; SHA256 `9ced640551dc90940fdc06ba65dceff02d590ce08aaf8e5af4d34d9c25b6c2f5`. Checkpoint manifest SHA256 `b4850eac1cc9b78d4ec4c0ec3cfc6b62de6f035b99bd321c10532549d57eb568`.
- Unpinned metric dependencies selected Transformers 5.20.2, whose own source requires Torch >=2.6. Preserve the official KARL Torch version and pin compatible older metric dependencies instead; evidence `reproduction/evidence/metric-compatibility.json`.
- CPU configuration job 2826692, source `79f29dd`, stage `/data/scratch-fast/kwen1/karl-reproduce-20261023-stages/20261023_221510-68884e-79f29dd-code`, output `/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/cpu-configuration-2826692/`. Owner remains `karl-reproduce-20261023`.
- No GPU has been requested. Strict model loading, installed-package source checks and metric-weight preparation must finish first.

### CPU checks complete; first GPU request

- Configuration job 2826692 completed 0:0 in 423 s, peak host RSS 5,641,148 KiB. Both strict inference loads and metric initialization passed; pinned metric dependencies retain Torch 2.3.0+cu121, and pip check passes.
- Checkpoint coverage is explicit in reproduction/evidence/checkpoint-inference-coverage.json: 833 EMA keys minus 440 training GAN-loss keys minus two obsolete, unreferenced conditioning tables = 391 strictly loaded inference keys. Encoder, decoder, both quantizers/base tokenizer, learned halting and the current rec_loss_embeddings are included. Evaluation never enters the training loss block.
- Input audit job 2826785 completed 0:0 in 170 s, peak RSS 3,963,604 KiB. All 5,000 JPEGs decode; 4,912 RGB and 88 grayscale inputs are explicitly converted to RGB. All 5,000 output stems are unique. 136 small images receive released CenterCrop padding. Ten metric weight files have frozen SHA256 values in cpu-input-audit-2826785/results.json.
- Proposed attended diagnosis: one RTX 2080 Ti (11 GiB), 4 CPUs, 16 GiB RAM, 30 minutes, vision-shared-rtx2080ti / vision-torralba-urops-meng / shared-if-available. Torch includes native sm_75. Model weights are about 1.08 GB plus metric models and activations; measure actual memory and runtime before final jobs. This allocation does not use B1's H100 resources. Avoid the separately reported faulty isola-3080-1 node.
- No reconstruction or paper result exists yet. Smoke runs will be labeled separately.

### Attended GPU diagnosis complete; full protocol frozen

- Single attended allocation 2826901 completed 0:0 and was released after 487 s on inspector-gadget.drl. All three steps completed 0:0; no GPU retry or model/metric fix was needed. Driver 580.173.02, RTX2080Ti, Torch 2.3.0+cu121/CUDA12.1, mounts verified.
- Both released checkpoints passed eight-image smoke tests at all eight fixed budgets and all three adaptive epsilons. Released PNG-based metrics work with direct 256-pixel inputs. Each smoke has its own report.md/results.json and exact full stage; summary: /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/gpu-diagnostic-2826901/.
- Reconstruction peak GPU allocation 10,015,640,576 bytes; default FID batch50 peak1,289,212,416 bytes. This is SMOKE_ONLY, not a paper result.
- Exact eleven-condition inventory, primary Table1/2 targets, separate additional Table4 reconstruction-FID interpretation, preprocessing/masking and metric-specific sample counts are frozen in reproduction/frozen-protocol.md and targets.json before full runs. Table4 does not explicitly identify training checkpoint or a separate sample count; use released evaluator pairing against all5000 GT images and report that limitation.
- Reuse official FID functions and identical ground-truth statistics across conditions; save each mean/covariance. No feature, estimator or formula change.
- Pre-final current account/partition/GPU/QoS inventory retained privately at karl-reproduction-results/resource-inventory-pre-final-20261023.json. B1 H100 jobs remain untouched. Final request: one2080Ti,4CPU,16GiB,6h for each released checkpoint; roughly3h/checkpoint estimated from warm smoke timing plus full scoring/I/O/FID.

### Full-split evaluation receipts

- Both final jobs use frozen source fd946c7 and the same complete 5,000-image manifest. No limited sample or smoke output is substituted. Each has one RTX2080Ti,4CPU,16GiB,6h, Vision Shared; owner remains karl-reproduce-20261023. Preflight must pass on the actual allocated node before inference.
- ImageNet1K checkpoint job2827023: stage /data/scratch-fast/kwen1/karl-reproduce-20261023-stages/20261023_224053-d730d6-fd946c7-code; output /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/final-2827023-imagenet; adjacent final-imagenet-2827023.slurm.log.
- ImageNet100 checkpoint job2827031: stage /data/scratch-fast/kwen1/karl-reproduce-20261023-stages/20261023_224128-cb7936-fd946c7-code; output /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/final-2827031-imagenet100; adjacent final-imagenet100-2827031.slurm.log.
- Exact receipts: karl-reproduction-results/final-submissions.json. Jobs are accepted, not completed; no paper result is inferred yet.

### Full evaluations terminal; CPU aggregation prerequisite fixed

- GPU jobs 2827023 and 2827031 completed 0:0 on inspector-gadget.drl in 8,512 and 8,464 seconds. Both results.json files are MEASURED. Their full sample counts and raw metric/token completeness still require independent readback checks.
- CPU aggregation job 2828148 stopped before scientific checks because Matplotlib was absent. Failed stage /data/scratch-fast/kwen1/karl-reproduce-20261023-stages/20261024_010826-bb8e7e-a6f91b6-code and karl-reproduction-results/aggregation-2828148 retain the operational failure report/log. No GPU result is classified as failed because of this dependency error.
- Install pinned Matplotlib 3.7.5 for CPU figures, retain NumPy 1.24.1, record the post-install analysis lock and require pip check. Retry only the CPU aggregation from a fresh full stage. No GPU resubmission, checkpoint selection or evaluation-setting change.

## 2026-10-24: released-checkpoint assignment completed

- Outcome: both full released-checkpoint evaluations and CPU completeness checks completed. Several values differ from the paper; a complete numerical reproduction is not established. No arbitrary paper-success threshold or unsupported confidence interval is used.
- ImageNet1K checkpoint job 2827023 completed 0:0 in 8,512 s; ImageNet100 checkpoint job 2827031 completed 0:0 in 8,464 s. Both ran on inspector-gadget.drl with one RTX2080Ti, 4 CPUs and 16 GiB each. Peak GPU allocated: 10,015,640,576 bytes each. Slurm peak host RSS: 10,698,184 / 10,778,596 KiB; separate process RSS values are retained in run results.
- Each checkpoint has 5,000 images at all eleven settings, 55,000 token records and 5,000 finite observations per metric/condition. Fixed active-token counts are exact. Checkpoint hashes still match their original receipts. CPU job 2828154 completed 0:0 in 261 s and independently verified arrays, counts, filenames and reported means.
- At 256 fixed tokens, IN1K actual L1 x10 / LPIPS / SSIM / DreamSim = .642737 / .148123 / .518763 / .129853; paper = .68 / .14 / .50 / .11. IN100 actual = .703254 / .138712 / .486422 / .107700; paper = .70 / .14 / .49 / .11. Full absolute differences for Tables 1 and 2 are retained in the consolidated report and results JSON.
- Adaptive mean tokens for epsilon .03/.05/.09: IN1K 252.0904 / 225.5046 / 146.7592 vs paper 252.1 / 225.2 / 146.2; IN100 252.9702 / 229.6596 / 152.4792 vs 252.9 / 229.6 / 152.5.
- Additional reconstruction FID at fixed 32/128/256: IN1K 33.574478 / 14.301705 / 11.215062; IN100 29.229618 / 13.894876 / 8.809673; paper Table 4 = 28.7 / 13.6 / 8.8. Table 4 does not identify its training-checkpoint association or a separate sample count. Both results are shown; the nearer result does not establish that association. No generation result is inferred.
- GT cache readback job 2828173 completed 0:0 in 283 s. All 5,000 paired GT PNGs are byte-identical. Independent GPU GT feature means/covariances differ by at most 1.4489619e-06 / 3.4794372e-06; swapping the caches changes the twelve checked FIDs by at most 4.079987e-05. This measured numerical sensitivity is not statistical uncertainty.
- Primary retained report/results: /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/aggregation-2828154/report.md and results.json. GT readback: /data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-results/gt-cache-readback-2828173/report.md and results.json. PNG/PDF figures in the primary report directory: fixed-token-comparison, adaptive-comparison and reconstruction-fid-comparison. All three figure layouts were inspected through read-only previews; retained files stay on CSAIL.
- Provenance index: reproduction/evidence/run-index.json. Artifact hashes: reproduction/evidence/final-artifact-receipts.json. Inference source fd946c7532bd6ac90d62dcb625f38bc45e6c6ec6; main aggregation source 2f03ae5; GT readback source b66abde. All model, base-tokenizer, training and released evaluator files remain identical to official source 5c1a5693edf87228f988f363b2acdb7eb826a889.
- Strict checkpoint coverage: 833 EMA entries = 440 GAN training-loss entries + 2 obsolete unused tables + 391 strictly loaded current inference entries. Encoder, decoder, base tokenizer, quantizers and halting weights are complete. Installed-package metric compatibility and official-weight identities are retained with the CPU evidence.
- Protocol limits remain explicit: repaired unfinished evaluator; requested epsilon 0 maps to .005; fixed masking is explicitly disabled; adaptive cutoff .75 differs from unused YAML .55; released standard-data cropping omits the demo resize and pads 136 small images; metric versions are not fully pinned by the paper. No post-result checkpoint/setting selection was used.
- Operational failures remain separate: one pre-submission staging permission failure and CPU aggregation 2828148 missing Matplotlib. Both are preserved. Matplotlib 3.7.5 with NumPy 1.24.1 fixed the CPU plotting prerequisite; pip check passed. The initial unpinned metric dependency mismatch was resolved on CPU before GPU use.
- Scope error: the unmentioned fork https://github.com/kw7243/dino was created at 2026-10-23T21:58:46Z. It and its plain local copy are preserved; no model or experiment used its code. The installed DreamSim implementation is used. No further repository was created.
- Training recommendation: resolve the IN1K high-budget evaluation-protocol discrepancy before full training. Literal released scripts specify 200 epochs at 8 x 64 images, then 400 epochs at 8 x 18, accum_iter 1. Existing split counts imply 49,400 + 351,600 loader iterations for IN100 (126,689 files), or 500,400 + 3,558,800 for IN1K (1,281,167 files). Conditional wall/GPU-hour formulas and the appendix's 8 x 80 GB GPU reference are in the consolidated results; no unsupported absolute training-time estimate is inferred from evaluation timing.
- Next action: scope a protocol-focused follow-up for the IN1K 256-token discrepancy. This bounded released-checkpoint assignment is complete. All owned jobs are terminal; active terminal and canonical notebook ownership return to Firstmate with the completion handoff.
