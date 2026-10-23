# Source setup notebook

## Current state

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
- No existing divergent branch was changed. No force push was used.

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
