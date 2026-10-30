"""Prepare local model configuration without requiring a scene-generation repo."""
import argparse
import hashlib
import json
from pathlib import Path

from .common import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(args):
    base = args.env_root.resolve()
    tools = base.parent / 'assetforge-tools'
    key = hashlib.sha256(str(args.config.resolve()).encode()).hexdigest()[:12]
    folder = tools / 'assetforge-deployments' / key / 'configs'
    folder.mkdir(parents=True, exist_ok=True)
    source = args.from_run / 'configs' if args.from_run else ROOT / 'Pipeline/Asset_gen'
    names = {'image': 'prompt_to_image_config.json', 'mask': 'image_to_mask_config.json', 'mesh': 'image_mask_to_3d_config.json'}
    configs = {name: read_json(source / (name + '.json' if args.from_run else filename)) for name, filename in names.items()}
    image, mask, mesh = (configs[name] for name in ('image', 'mask', 'mesh'))
    if args.flux_model:
        image['model_path'] = str(args.flux_model.resolve())
    if args.sam3_model:
        mask['sam3']['checkpoint_path'] = str(args.sam3_model.resolve())
    if args.sam_model:
        mask['sam']['checkpoint_path'] = str(args.sam_model.resolve())
    pipeline = args.sam3d_pipeline or (Path(mesh['pipeline_config_path']) if args.from_run else None)
    if not pipeline or not pipeline.is_file():
        raise ValueError('Supply --sam3d-pipeline /local/checkpoints/pipeline.yaml, or adopt --from-run /successful/run')
    for name, path in [('FLUX.1', image['model_path']), ('SAM3', mask['sam3']['checkpoint_path'])]:
        if not Path(path).is_dir():
            raise ValueError(name + ' local model directory missing: ' + str(path))
    import yaml
    pipeline = pipeline.resolve()
    def absolute(value, key=''):
        if isinstance(value, dict):
            return {k: absolute(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [absolute(v, key) for v in value]
        if isinstance(value, str) and (key.endswith(('_path', '_dir')) or key == 'pretrained_model_name_or_path'):
            candidate = pipeline.parent / value
            if candidate.exists():
                if key == 'pretrained_model_name_or_path' and candidate.is_dir() and (candidate / 'model.pt').is_file():
                    candidate = candidate / 'model.pt'
                return str(candidate.resolve())
        return value
    data = absolute(yaml.safe_load(pipeline.read_text()))
    data.update(rendering_engine='pytorch3d', decode_formats=['gaussian', 'mesh'])
    (folder / 'pipeline.yaml').write_text(yaml.safe_dump(data))
    cache = tools / 'assetforge-hf-cache'
    cache.mkdir(exist_ok=True)
    image.update(cache_path=str(cache), code_path=None, memory_strategy='model_cpu_offload', prompt_suffix='')
    image['inference_options'].update(height=1024, width=1024, num_inference_steps=50, guidance_scale=3.5, max_sequence_length=512)
    for name in ('sam', 'sam3'):
        mask[name].update(enabled=name == 'sam3', isolate_process=False)
    mask['sam']['mask_combination'] = 'union'
    mesh.update(repository_path=str(tools / 'sam-3d-objects'),
                inference_module_path=str(tools / 'sam-3d-objects/notebook/inference.py'),
                pipeline_config_path=str(folder / 'pipeline.yaml'), isolate_process=False, export_glb=True, compile_model=False)
    for name, value in configs.items():
        write_json(folder / (name + '.json'), value)
    deployment = {'schema_version': 1, 'env_root': str(base), 'tools': str(tools),
                  'blender': (str(args.blender.resolve()) if args.blender else
                              str(tools / 'blender/blender') if (tools / 'blender/blender').is_file() else None),
                  'configs': {name: str(folder / (name + '.json')) for name in configs}}
    write_json(args.config.resolve(), deployment)
    print('Deployment config:', args.config.resolve(), flush=True)
    return deployment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    for name in ('from-run', 'flux-model', 'sam3-model', 'sam-model', 'sam3d-pipeline', 'blender'):
        p.add_argument('--' + name, type=Path)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
