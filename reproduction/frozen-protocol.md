# Released-checkpoint evaluation protocol

Frozen before full-split evaluation, 2026-10-23 UTC. Source 5c1a5693edf87228f988f363b2acdb7eb826a889. Exact values and the complete eleven-condition inventory are in targets.json.

Both released quantized VQGAN EMA checkpoints are evaluated separately: ImageNet1K-trained and ImageNet100-trained. Their measured hashes are in cpu-prepare-2826204/checkpoints.json and the coverage proof. No checkpoint selection is based on smoke quality.

Use the same 5,000 ImageNet100 validation images for every condition and metric, 50 per source-defined synset. Manifest SHA256: 9ced640551dc90940fdc06ba65dceff02d590ce08aaf8e5af4d34d9c25b6c2f5. All filenames, hashes, labels and source paths are retained. Order is the frozen manifest order; no shuffle or sampling. RGB conversion, CenterCrop(256), ToTensor(), no resize, normalization or augmentation. The 136 small inputs are padded by released CenterCrop. Resolution is always 256 x 256. Batch 1, FP32, seed 0, source cuDNN benchmarking, matmul TF32 off, cuDNN TF32 on. No repeated-seed variance is claimed.

| Condition | Budget | Requested epsilon | Effective bin | Mask cutoff | Primary paper target | Additional comparison |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| fixed_32 | 32 | 0 | .005 | 1.0 | Table 1 | Table 4 reconstruction FID |
| fixed_64 | 64 | 0 | .005 | 1.0 | No exact paper number | Released-evaluator curve point |
| fixed_96 | 96 | 0 | .005 | 1.0 | No exact paper number | Released-evaluator curve point |
| fixed_128 | 128 | 0 | .005 | 1.0 | Table 1 | Table 4 reconstruction FID |
| fixed_160 | 160 | 0 | .005 | 1.0 | No exact paper number | Released-evaluator curve point |
| fixed_192 | 192 | 0 | .005 | 1.0 | No exact paper number | Released-evaluator curve point |
| fixed_224 | 224 | 0 | .005 | 1.0 | No exact paper number | Released-evaluator curve point |
| fixed_256 | 256 | 0 | .005 | 1.0 | Table 1 | Table 4 reconstruction FID |
| adaptive_0.03 | 256 | .03 | .03 | .75 | Table 2 | Table 4 reconstruction FID |
| adaptive_0.05 | 256 | .05 | .05 | .75 | Table 2 | Table 4 reconstruction FID |
| adaptive_0.09 | 256 | .09 | .09 | .75 | Table 2 | Table 4 reconstruction FID |

The model masks probabilities strictly greater than the cutoff. Fixed cutoff 1.0 explicitly retains every token, as required by the paper same-count regime. Released epsilon=0 alone does not disable masking and maps to the lowest conditioning bin .005. Adaptive inference keeps the actual released .75 cutoff, not the unused YAML .55. Exact requested adaptive epsilons are present in the released conditioning bins. There is no iterative reconstruction-quality search or sample rejection. Record every actual active-token count.

Use released save_imgs: clipped output multiplied by 255 and truncated to uint8 PNG; reload paired PNGs as RGB tensors. L1 is the per-image mean absolute pixel difference, averaged over 5,000 pairs; multiply by 10 only for paper table display. LPIPS uses AlexNet and source default normalize=False with [0,1] inputs. SSIM and PSNR use the released TorchMetrics classes with data_range=1.0; average per-image outputs. DreamSim uses released direct 256-pixel PNG tensors, without its returned preprocessing function. No transform correction is silently applied. Each metric and mean-token statistic has exactly 5,000 image observations per checkpoint and condition. PSNR and mean-square pixel error have no numerical paper targets.

Primary Table 1 comparisons are L1 x10, LPIPS, SSIM and DreamSim at 32/128/256 tokens, using the appropriate training-checkpoint row. Primary Table 2 comparisons are mean active tokens, LPIPS and SSIM at epsilon .03/.05/.09, also with the matching row. Additional fixed-budget points and L1/DreamSim/PSNR for adaptive runs are reported separately, with no invented paper value. No architecture scaling figure or training ablation is claimed from these two checkpoints.

Additional reconstruction FID uses official pytorch-fid 2048-dimensional Inception features, default batch 50, over 5,000 reconstructions versus the same 5,000 paired ground-truth PNGs. Table 4 reports IN100 but does not separately specify its sample count or training-checkpoint association. The released evaluator compares reconstructed folders against their paired ground-truth folder, so this run uses the frozen full validation split. Report this interpretation and both checkpoint results, without silently choosing the nearer one. FID is a distribution statistic, separate from per-image reconstruction metrics and token counts. No generation FID or generation claim is made.

Paper uncertainty is unreported. Two-decimal metric display has rounding precision +/- .005; one-decimal token display has precision +/- .05. These are not statistical confidence intervals or success thresholds. Show actual value, paper value and absolute difference. Label the attended small-sample runs SMOKE_ONLY.

The official FID functions compute the identical paired ground-truth statistics once per checkpoint run and reuse them across conditions. Reconstructions each have separate statistics. This changes execution reuse only, without changing files, features, covariance estimator or distance formula. Statistics are retained.
