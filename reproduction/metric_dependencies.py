"""Use DreamSim's pinned, bundled DINO implementation and existing base weights.

No GitHub/Torch Hub repository is imported, fetched, or executed here. All forward
methods and the vit_base constructor in DreamSim 0.2.1 match the DINO source AST.
Only initialization differs; strict full state loading replaces every such value.
"""
from pathlib import Path
import hashlib, json, importlib.metadata
import torch
DINO_WEIGHTS=Path('/data/vision/torralba/u/kwen1/image-tokenization-20261023/karl-reproduction-assets/dino-base-pretrained.pth')
def pin_dino_source():
    from dreamsim.feature_extraction import vision_transformer as vt
    assert importlib.metadata.version('dreamsim')=='0.2.1'
    evidence=json.loads(Path('reproduction/evidence/dino-package-equivalence.json').read_text())
    assert hashlib.sha256(Path(vt.__file__).read_bytes()).hexdigest()==evidence['packaged_module_sha256']
    vit_base=vt.vit_base
    def load(repo_or_dir,model,*args,**kwargs):
        if repo_or_dir=='facebookresearch/dino:main':
            assert model=='dino_vitb16' and not args and not kwargs
            result=vit_base(patch_size=16,num_classes=0)
            result.load_state_dict(torch.load(DINO_WEIGHTS,map_location='cpu',weights_only=True),strict=True)
            return result
        raise RuntimeError('Unexpected dynamic Torch Hub repository request: '+str(repo_or_dir))
    torch.hub.load=load
