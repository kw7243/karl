"""Use DreamSim's pinned, bundled DINO implementation and existing base weights.

No GitHub/Torch Hub repository is imported, fetched, or executed here. All forward
methods and the vit_base constructor in DreamSim 0.2.1 match the DINO source AST.
Only initialization differs; strict full state loading replaces every such value.
"""
from pathlib import Path
import torch
DINO_WEIGHTS=Path('/data/scratch-fast/kwen1/.cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth')
def pin_dino_source():
    from dreamsim.feature_extraction.vision_transformer import vit_base
    original=torch.hub.load
    def load(repo_or_dir,model,*args,**kwargs):
        if repo_or_dir=='facebookresearch/dino:main':
            assert model=='dino_vitb16' and not args and not kwargs
            result=vit_base(patch_size=16,num_classes=0)
            result.load_state_dict(torch.load(DINO_WEIGHTS,map_location='cpu',weights_only=True),strict=True)
            return result
        raise RuntimeError('Unexpected dynamic Torch Hub repository request: '+str(repo_or_dir))
    torch.hub.load=load
