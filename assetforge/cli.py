"""Unified asset generation, comparison and scratch setup CLI."""
import argparse
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
CUP = ('a single yellow ceramic mug with one handle, fully visible, isolated on a plain white background, '
       'product photography, sharp focus, realistic materials, no depth of field blur')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--setup', action='store_true')
    mode.add_argument('--test', nargs='?', const=CUP, metavar='PROMPT', help='Four-cell comparison; default subject is a cup')
    mode.add_argument('--prompt')
    mode.add_argument('--image', type=Path)
    p.add_argument('--image-backend', choices=['flux', 'flux2'], default='flux')
    p.add_argument('--mask-backend', choices=['sam3', 'sam'], default='sam3')
    p.add_argument('--mesh-backend', choices=['sam3d', 'trellis2'], default='sam3d')
    p.add_argument('--mask-prompts', nargs='+', metavar='NOUN')
    p.add_argument('--mask', type=Path, help='Use an existing foreground mask instead of segmentation')
    p.add_argument('--label', help='Uppercase asset label; used in filenames and camera presets')
    p.add_argument('--seed', type=int, default=314)
    p.add_argument('--output', type=Path)
    p.add_argument('--config', type=Path, default=Path(os.environ.get('ASSETFORGE_CONFIG', ROOT / 'assetforge.local.json')))
    previews = p.add_mutually_exclusive_group()
    previews.add_argument('--render', action='store_true', help='Render studio previews; requires optional Blender')
    previews.add_argument('--no-render', action='store_true', help='Skip Blender previews in comparison mode')
    p.add_argument('--budget', type=int, default=3300)
    p.add_argument('--flux2-device-strategy', choices=['balanced', 'cpu_offload'], default='balanced')
    p.add_argument('--install-dir', type=Path, help='Installation directory; defaults to SCRATCH or ~/.local/share/assetforge')
    p.add_argument('--env-root', type=Path, help='Environment prefix inside the installation directory')
    p.add_argument('--from-run', type=Path, help='Setup: adopt compatible model configs from a successful prior run')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--with-extras', action='store_true', help='Setup optional FLUX.2 and TRELLIS.2 environments/models')
    p.add_argument('--environments-only', action='store_true', help='Setup base runtime first, before model login/downloads')
    p.add_argument('--flux2-variant', choices=['dev', 'klein'], default='dev')
    p.add_argument('--flux-model', type=Path)
    p.add_argument('--sam3-model', type=Path)
    p.add_argument('--sam-model', type=Path)
    p.add_argument('--sam3d-pipeline', type=Path)
    visualization = p.add_mutually_exclusive_group()
    visualization.add_argument('--with-blender', action='store_true', help='Setup: download optional Blender for previews')
    visualization.add_argument('--blender', type=Path, help='Setup: use an existing Blender executable')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.setup:
        from .installation import install
        try:
            install(a)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        return
    if not a.prompt and a.test is None and not a.image:
        p.error('Choose --prompt, --image, --test [PROMPT], or --setup')
    if a.mask and a.test is not None:
        p.error('--test generates two different images; supply mask queries instead of one shared mask file')
    if not a.mask_prompts:
        a.mask_prompts = ['mug'] if a.test == CUP else [a.prompt or a.test or a.image.stem]
    if a.image and not a.image.is_file():
        p.error('--image must point to an existing image')
    if a.mask and not a.mask.is_file():
        p.error('--mask must point to an existing mask')
    a.label = a.label or ('CUP' if a.test == CUP else 'ASSET')
    if not re.fullmatch(r'[A-Z][A-Z0-9_]*', a.label):
        p.error('--label must contain uppercase letters, digits and underscores')
    if a.budget <= 0:
        p.error('--budget must be positive')
    if not a.output:
        a.output = Path(os.environ.get('SCRATCH', 'runs')) / 'Asset-forge' / (
            time.strftime('%Y%m%d-%H%M%S') + '-' + os.environ.get('SLURM_JOB_ID', 'local'))
    from .runtime import generate
    try:
        code = generate(a)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    raise SystemExit(code)
