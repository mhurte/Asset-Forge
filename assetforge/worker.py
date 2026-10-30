"""One model stage per process, using the existing tested stage implementations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Pipeline/Asset_gen'))


def masks(request, generator):
    from PIL import Image, ImageChops
    backend = request['backend']
    prompts = request['mask_prompts'] if backend == 'sam3' else [None]
    components, union = [], None
    for index, prompt in enumerate(prompts):
        directory = Path(request['directory']) / ('mask-part-' + str(index))
        directory.mkdir(parents=True, exist_ok=True)
        result = generator(request['image'], prompt, config_path=request['config'], output_dir=directory, backend=backend)
        path = Path(result.mask_paths[backend])
        with Image.open(path) as source:
            mask = source.convert('L')
        if not mask.getbbox() or (union is not None and union.size != mask.size):
            raise ValueError('Empty or inconsistent participant mask: ' + str(prompt))
        union = mask if union is None else ImageChops.lighter(union, mask)
        components.append({'prompt': prompt, 'path': str(path)})
    target = Path(request['directory']) / 'foreground-mask.png'
    union.save(target)
    return {'mask': str(target), 'components': components, 'backend': backend}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['image', 'mask', 'mesh'])
    p.add_argument('request', type=Path)
    a = p.parse_args()
    r = json.loads(a.request.read_text())
    if a.stage == 'image':
        from prompt_to_image import generate_image
        output = {'image': str(generate_image(r['prompt'], config_path=r['config'], output_path=r['image'], seed=r['seed']).image_path)}
    elif a.stage == 'mask':
        from image_to_mask import generate_mask
        output = masks(r, generate_mask)
    else:
        from assetforge.offline import offline_torch_hub
        from image_mask_to_3d import generate_3d_asset
        with offline_torch_hub(Path(r['tools'])):
            result = generate_3d_asset(r['image'], r['mask'], config_path=r['config'], output_path=r['ply'],
                                       seed=r['seed'], _allow_isolation=False)
        if result.mesh_path is None:
            raise RuntimeError('SAM3D must export both GLB and Gaussian PLY')
        output = {'glb': str(result.mesh_path), 'ply': str(result.asset_path)}
    Path(r['result']).write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
