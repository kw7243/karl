# A1 source audit

## Seed use

The recorded seed remains 0; it is never relabeled 20261023.
The evaluated VQGAN path has no RNG draw after strict checkpoint loading and model.eval().
Model construction initializes weights randomly, then load_inference_state strictly loads all 391 current inference state entries.
The 440 GAN-loss entries and two obsolete, unused tables are explicitly excluded; no missing inference entry is accepted.
In sample_token_counts, randperm is under self.training; multinomial is under sampled_code_losses is None.
The driver supplies each request as a Python float and sets model.eval(), so both draws are bypassed.
get_discretized_loss_and_embedding is called without token_add_count; its two random augmentation branches are bypassed.
The .05/.09 requests select the corresponding fixed loss embeddings.
KARL VectorQuantizer uses distance argmin, not sampling; EMA update is training-only.
VQGANWrapper selects VQModel, not the VAE posterior.sample path.
VQModel.encode uses its deterministic convolutional encoder and VectorQuantizer2 argmin.
VectorQuantizer2 remap defaults to None; unknown-index randint is therefore unreachable.
The base VQGAN config sets dropout0; model.eval also disables dropout.
MultiheadAttention uses its default dropout0; no token noise or random masking occurs in eval.
Preprocessing is RGB and CenterCrop/ToTensor, with fixed ordered manifest traversal.
Released LPIPS is eval(), SSIM is deterministic, and scoring has no seeded sampling.
Thus seed0 versus20261023 does not select different scientific inputs or an RNG stream for these two reference conditions.
This is a reachable-source proof, not a paired-seed GPU measurement or a claim of bitwise GPU determinism.
cuDNN benchmark/kernel nondeterminism remains separate from random seed semantics.

## Slot identities and native batch1

forward_executer uses latent_tokens[:256] and decoder_latent_tokens_timestep_embed[:256] in their original order.
prepare_decoder_latents constructs halt_prob>0.75 for each original position, builds row/column attention exclusion, clears its diagonal, and removes only the leading loss-conditioning position.
The decoder adds the matching original positional embedding before concatenation; neither adapter nor source sorts, compacts, or replaces the active mask with a prefix.
The adapter changes only halting_prob (1.0 fixed, .75 adaptive) and records active counts.
For batch1, mask.repeat(n_head,1,1) correctly supplies one image mask to all heads.
For batch>1, repeat(n_head,1,1) orders images incorrectly for PyTorch batch-major/head-minor masks; larger-batch equivalence is not claimed.
All reused inference and source perceptual scoring use batch1, so that unresolved larger-batch path is outside this result.
No per-image probability/mask vectors were retained.
Preserved original-slot execution is proved from the byte-verified executed source; individual active slot lists and evidence that a particular image uses a nonprefix mask are unavailable and are not inferred from counts.

## Source L1 versus request and source gate

0.75 is the halt-probability threshold, not image reconstruction error.
The request is a learned conditioning value, not a post-hoc error-enforcement rule.
source-inference.log prints raw reconstruction L1 before saving; its nested image/condition loop has exactly one line per output.
Pairing will be verified against all 55,000 ordered token records, condition budgets, active counts, and the 5,000-image manifest.
PNG loglaplace is a different, clipped/truncated observable; Table1 x10 scaling does not apply to achieved image L1.
The A1 pass gate constrains token reduction/counts and source LPIPS, not a bound on the violation rate.
