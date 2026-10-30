"""Optional experimental backends. Existing SAM3D/FLUX.1 entry points stay unchanged.

Execute each call in its own matching environment; models are never co-resident.
"""
import json
from pathlib import Path
import sys


def generate_flux2(prompt, model_path, output_path, *, variant='dev', seed=42,
                   resolution=1024, steps=None, device_strategy='cpu_offload'):
    import torch
    from diffusers import Flux2Pipeline, Flux2KleinPipeline
    if variant not in {'dev', 'klein'}:
        raise ValueError('FLUX.2 variant must be dev or klein')
    if device_strategy not in {'cpu_offload', 'balanced'}:
        raise ValueError('device_strategy must be cpu_offload or balanced')
    cls = Flux2Pipeline if variant == 'dev' else Flux2KleinPipeline
    options = {'torch_dtype': torch.bfloat16, 'local_files_only': True, 'low_cpu_mem_usage': True}
    if device_strategy == 'balanced':
        if torch.cuda.device_count() < 2:
            raise ValueError('Balanced comparison requires at least two visible GPUs; submit the updated two-GPU job')
        # Leave activation/decoder headroom on each GPU instead of filling VRAM with weights.
        options.update(device_map='balanced', max_memory={i: torch.cuda.get_device_properties(i).total_memory
                                                         - 8 * 1024**3 for i in (0, 1)})
        if min(options['max_memory'].values()) <= 0:
            raise ValueError('Insufficient GPU memory for balanced placement')
    pipeline = cls.from_pretrained(str(model_path), **options)
    mapping = getattr(pipeline, 'hf_device_map', None) if device_strategy == 'balanced' else None
    if device_strategy == 'cpu_offload':
        pipeline.enable_model_cpu_offload()
    else:
        print('FLUX.2 device map:', mapping, flush=True)
        devices = {str(device).removeprefix('cuda:') for device in (mapping or {}).values()}
        if devices != {'0', '1'}:
            raise RuntimeError('Expected FLUX.2 components on both GPUs without CPU/disk spill; got ' + str(mapping))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix('.placement.json').write_text(json.dumps(
        {'strategy': device_strategy, 'device_map': mapping, 'precision': 'bfloat16'}, indent=2))
    image = pipeline(prompt=prompt, height=resolution, width=resolution,
                     num_inference_steps=steps or (50 if variant == 'dev' else 4),
                     guidance_scale=2.5 if variant == 'dev' else 1.0,
                     generator=torch.Generator('cpu').manual_seed(seed)).images[0]
    image.save(destination)
    return destination


def generate_trellis2(image_path, mask_path, output_path, *, repository_path,
                     model_path, seed=42, pipeline_type='1024_cascade', texture_size=2048):
    """Reconstruct using the SAME externally supplied mask as SAM3D.

    The prepared pipeline selects ProvidedMask instead of downloading a second
    segmentation model. Its callable fails if a mask is accidentally discarded.
    No upstream source files are modified.
    """
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    sys.path.insert(0, str(repository_path))
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg
    import o_voxel

    class ProvidedMask:
        def __init__(self, **kwargs):
            pass
        def to(self, *args):
            pass
        def cpu(self):
            pass
        def __call__(self, image):
            raise ValueError('TRELLIS comparison requires a nontrivial supplied alpha mask')

    rembg.ProvidedMask = ProvidedMask
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
    pipeline.cuda()
    with Image.open(image_path) as source, Image.open(mask_path) as mask_source:
        image = source.convert('RGBA')
        mask = mask_source.convert('L')
        if mask.size != image.size or not mask.getbbox() or mask.getextrema() == (255, 255):
            raise ValueError('Mask must match the image, contain foreground, and exclude background')
        image.putalpha(mask)
    mesh = pipeline.run(image, seed=seed, pipeline_type=pipeline_type)[0]
    mesh.simplify(16777216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=500000, texture_size=texture_size, remesh=True,
        remesh_band=1, remesh_project=0, verbose=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Standard embedded PNG/JPEG textures, preserving metallic/roughness/alpha.
    glb.export(str(destination), extension_webp=False)
    if not destination.is_file() or destination.stat().st_size < 20:
        raise RuntimeError('TRELLIS did not export a nonempty GLB')
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['flux2', 'trellis2'])
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    function = generate_flux2 if args.backend == 'flux2' else generate_trellis2
    path = function(**request)
    print(json.dumps({'output': str(path)}), flush=True)
