"""Strictly load every current inference parameter from the released EMA state."""
# These checkpoint-only conditioning tables have no references in current code.
LEGACY_TRAINING_KEYS={'code_loss_embeddings','nll_loss_embeddings'}
def load_inference_state(model,weights):
    ignored={k for k in weights if k.startswith('gan_losses.') or k in LEGACY_TRAINING_KEYS}
    model.load_state_dict({k:v for k,v in weights.items() if k not in ignored},strict=True)
    return sorted(ignored)
