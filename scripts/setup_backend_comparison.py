"""Online, GPU-less setup for isolated FLUX.2 and TRELLIS.2 experiment prefixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'trellis2': ('https://github.com/microsoft/TRELLIS.2.git', '75fbf0183001ed9876c8dbb35de6b68552ee08bd'),
    'diffusers': ('https://github.com/huggingface/diffusers.git', 'aae97e79e8e6a55e0b43114f9c8916c06d4bca8c'),
    'cumesh': ('https://github.com/JeffreyXiang/CuMesh.git', '12289e1062f0603f2f0d0771b02e1395d247f26f'),
    'flexgemm': ('https://github.com/JeffreyXiang/FlexGEMM.git', '6dd94a859c26ee8246888502eada3dd8ad85532e'),
    'nvdiffrast': ('https://github.com/NVlabs/nvdiffrast.git', '253ac4fcea7de5f396371124af597e6cc957bfae'),
    'utils3d': ('https://github.com/EasternJournalist/utils3d.git', '9a4eb15e4021b67b12c460c7057d642626897ec8'),
}
# This prior recipe has identical dependencies; only front-node validation changes.
COMPATIBLE_RECIPES = {'4a3ec45dfc0346baa5e2409824630b5b61a605e0e332fb9de15a4b371fbd9a0a', 'a1edc4c6b7a302cb6d2acf02b4d60dc7308bf8e4f60bc057da7a4c8676d46f8f'}
CPU_PROBES = {
    'flux2': 'from diffusers import Flux2Pipeline, Flux2KleinPipeline',
    'trellis2': '''import importlib.metadata as metadata
import torch, flash_attn, cumesh
from transformers import DINOv3ViTModel
for package in ('flex_gemm', 'o_voxel', 'nvdiffrast'):
    files = metadata.files(package)
    assert files and any(str(f).endswith('.so') and f.locate().is_file() for f in files), 'Missing compiled extension: ' + package
print('CPU imports and installed extension files checked; Triton/GPU imports deferred to allocated GPU job')
''',
}


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def finish_setup(target, prefixes, mamba, recipe):
    """No installs/downloads: inspect CPU-safe imports and leave GPU checks pending."""
    manifest = json.loads((target / 'models.json').read_text())
    if not (Path(manifest['flux2']) / 'model_index.json').is_file():
        raise FileNotFoundError('FLUX.2 download is incomplete; use normal --setup --resume')
    pipeline = json.loads((Path(manifest['trellis2']) / 'pipeline.json').read_text())
    for stem in pipeline['args']['models'].values():
        if not all(Path(stem + suffix).is_file() for suffix in ('.json', '.safetensors')):
            raise FileNotFoundError('Missing TRELLIS checkpoint: ' + stem)
    if not (Path(pipeline['args']['image_cond_model']['args']['model_name']) / 'config.json').is_file():
        raise FileNotFoundError('DINOv3 download is incomplete')
    revisions = {}
    sources = {name: target / name for name in SOURCES}
    for name, folder in sources.items():
        revisions[name] = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        if revisions[name] != SOURCES[name][1]:
            raise ValueError('Unexpected source revision: ' + name)
    for name, code in CPU_PROBES.items():
        run([mamba, 'run', '-p', prefixes[name], 'python', '-c', code])
        run([mamba, 'run', '-p', prefixes[name], 'python', '-m', 'pip', 'check'])
        with (target / (name + '-freeze.txt')).open('w') as output:
            run([mamba, 'run', '-p', prefixes[name], 'python', '-m', 'pip', 'freeze'], stdout=output)
    manifest.update(prefixes={k: str(v) for k, v in prefixes.items()}, sources={k: str(v) for k, v in sources.items()},
                    revisions=revisions, recipe=recipe, validation={'cpu_setup': 'passed', 'gpu_imports': 'pending'})
    temporary = target / 'ready.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(target / 'ready.json')
    print('Experimental CPU setup checks passed. Full TRELLIS/Triton imports run on the allocated GPU before generation.', flush=True)


def finalize_existing(target, prefixes, mamba, recipe):
    """Accept only the exact old recipe or this validation-only update."""
    markers = [target / 'recipe.sha256', *[p.with_name(p.name + '.backend-recipe') for p in prefixes.values()]]
    for marker in markers:
        if not marker.is_file() or marker.read_text() not in {recipe, *COMPATIBLE_RECIPES}:
            raise ValueError('Refusing unrecognized recipe marker: ' + str(marker))
    for prefix in prefixes.values():
        if not (prefix / 'bin/python').is_file():
            raise FileNotFoundError('Incomplete environment: ' + str(prefix))
    (target / 'ready.json').unlink(missing_ok=True)
    finish_setup(target, prefixes, mamba, recipe)
    for marker in markers:
        marker.write_text(recipe)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-root', type=Path, required=True, help='Existing level0 environment root')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--finalize-only', action='store_true', help='With --resume: verify existing installs/downloads without rebuilding')
    p.add_argument('--flux2', choices=['dev', 'klein'], default='dev')
    p.add_argument('--source-run', type=Path, help='Working run used to prepare SAM3D offline DINO dependencies')
    a = p.parse_args()
    if platform.system() != 'Linux':
        p.error('This setup requires Linux; model execution is validated on A100 only')
    base = a.env_root.resolve()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ.pop(key, None)
    os.environ['PYTHONNOUSERSITE'] = '1'
    scratch = Path(os.environ['SCRATCH']).resolve()
    if scratch != base.parent and scratch not in base.parents:
        p.error('The managed environment root must be inside SCRATCH')
    if a.finalize_only:
        if not a.resume:
            p.error('--finalize-only requires --resume')
        tools = base.parent / 'assetforge-tools'
        if not (base / 'setup-complete.json').is_file() or not (tools / 'offline-dependencies.json').is_file():
            p.error('Base setup/offline dependencies missing; use normal --setup --resume')
        prefixes = {name: base.with_name(base.name + '_' + name) for name in ('flux2', 'trellis2')}
        finalize_existing(tools / 'backend-comparison', prefixes, tools / 'micromamba/bin/micromamba',
                          hashlib.sha256((Path(__file__).read_bytes() + os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0").encode())).hexdigest())
        return
    if not (base / 'setup-complete.json').is_file():
        command = [sys.executable, ROOT / 'scripts/setup_level0.py', '--env-root', base]
        if a.resume:
            command.append('--resume')
        run(command)
    tools = base.parent / 'assetforge-tools'
    target = tools / 'backend-comparison'
    target.mkdir(parents=True, exist_ok=True)
    os.environ['PIP_CACHE_DIR'] = str(target / 'pip-cache')
    marker = target / 'recipe.sha256'
    recipe = hashlib.sha256((Path(__file__).read_bytes() + os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0").encode())).hexdigest()
    if marker.exists() and marker.read_text() != recipe and not (a.resume and marker.read_text() in COMPATIBLE_RECIPES):
        p.error('Experiment recipe changed; choose a different ASSETFORGE_ENV_ROOT rather than mixing builds')
    marker.write_text(recipe)
    (target / 'ready.json').unlink(missing_ok=True)
    if a.source_run:
        run([base.with_name(base.name + '_sam3d') / 'bin/python', '-B', '-m',
             'assetforge.offline', '--tools', tools, '--run', a.source_run],
            env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONNOUSERSITE': '1'})
    elif not (tools / 'offline-dependencies.json').is_file():
        p.error('SAM3D offline dependencies are missing; pass --source-run pointing to the known working run')
    mamba = tools / 'micromamba/bin/micromamba'
    if not mamba.is_file():
        p.error('Expected micromamba from level0 setup at ' + str(mamba))
    prefixes = {name: base.with_name(base.name + '_' + name) for name in ('flux2', 'trellis2')}
    for name, prefix in prefixes.items():
        owner = prefix.with_name(prefix.name + '.backend-recipe')
        if prefix.exists() and (not a.resume or not owner.exists() or owner.read_text() not in {recipe, *COMPATIBLE_RECIPES}):
            p.error('Existing prefix requires --resume and a matching ownership marker: ' + str(prefix))
        owner.write_text(recipe)
        if not (prefix / 'bin/python').exists():
            packages = ['python=3.11', 'pip', 'ninja', 'cmake=3.31']
            if name == 'trellis2':
                packages += ['cuda-toolkit=12.4', 'gcc_linux-64=12', 'gxx_linux-64=12']
            run([mamba, 'create', '-y', '-p', prefix, '-c', 'conda-forge', '-c', 'nvidia', *packages])
    sources = {}
    revisions = {}
    for name, (url, revision) in SOURCES.items():
        folder = target / name
        if not folder.exists():
            run(['git', 'clone', '--no-checkout', url, folder])
        run(['git', '-C', folder, 'checkout', '--detach', revision])
        run(['git', '-C', folder, 'submodule', 'update', '--init', '--recursive'])
        revisions[name] = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        sources[name] = folder
    def install(name, *packages, env=None):
        run([mamba, 'run', '-p', prefixes[name], 'python', '-m', 'pip', 'install', *packages], env=env)
    install('flux2', 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', 'https://download.pytorch.org/whl/cu126')
    install('flux2', sources['diffusers'], 'transformers==5.0.0', 'accelerate==1.12.0',
            'sentencepiece==0.2.1', 'protobuf==6.33.2', 'Pillow==11.3.0')
    install('trellis2', 'torch==2.6.0', 'torchvision==0.21.0', '--index-url', 'https://download.pytorch.org/whl/cu124')
    install('trellis2', 'setuptools==75.8.0', 'wheel==0.45.1', 'numpy==1.26.4',
            'transformers==4.57.3', 'timm==1.0.22', 'Pillow==11.3.0', 'opencv-python-headless==4.11.0.86',
            'imageio==2.37.2', 'imageio-ffmpeg==0.6.0', 'easydict==1.13', 'trimesh==4.10.1',
            'kornia==0.8.2', 'zstandard==0.25.0', 'tqdm==4.67.1', 'scipy==1.15.3',
            'networkx==3.4.2', 'einops==0.8.1', sources['utils3d'])
    env = os.environ.copy()
    prefix = prefixes['trellis2']
    env.update(CUDA_HOME=str(prefix), FORCE_CUDA='1', BUILD_TARGET='cuda', TORCH_CUDA_ARCH_LIST=os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0'),
               MAX_JOBS=os.environ.get('MAX_JOBS', '1'), CC=str(prefix / 'bin/x86_64-conda-linux-gnu-cc'),
               CXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from setup_level0 import configure_cuda_paths
    configure_cuda_paths(prefix, env)
    probe = target / 'cuda-build-probe.cpp'
    probe.write_text('#include <cuda_runtime_api.h>\nint main(){int v=0; return cudaRuntimeGetVersion(&v);}\n')
    run([env['CXX'], probe, '-lcudart', '-o', target / 'cuda-build-probe'], env=env)
    install('trellis2', 'flash-attn==2.7.3', '--no-build-isolation', '--no-deps', env=env)
    for folder in (sources['nvdiffrast'], sources['cumesh'], sources['flexgemm'], sources['trellis2'] / 'o-voxel'):
        install('trellis2', folder, '--no-build-isolation', '--no-deps', env=env)
    # Downloads run with the new image environment; resolve immutable HF snapshots.
    helper = target / 'download-models.py'
    helper.write_text(DOWNLOAD_HELPER)
    run([mamba, 'run', '-p', prefixes['flux2'], 'python', helper, target, a.flux2])
    finish_setup(target, prefixes, mamba, recipe)


DOWNLOAD_HELPER = r'''
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, model_info
target = Path(sys.argv[1])
flux_repo = 'black-forest-labs/FLUX.2-dev' if sys.argv[2] == 'dev' else 'black-forest-labs/FLUX.2-klein-4B'
revisions = {}
def download(repo, patterns=None):
    revision = model_info(repo).sha
    revisions[repo] = revision
    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns, cache_dir=str(target / 'hf-cache')))
flux = download(flux_repo, ['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '**/*.json', '**/*.safetensors', '**/*.model', '**/*.txt', '**/*.jinja'])
trellis = download('microsoft/TRELLIS.2-4B')
pipeline = json.loads((trellis / 'pipeline.json').read_text())
for key, value in pipeline['args']['models'].items():
    if value.startswith('microsoft/'):
        pieces = value.split('/')
        repo, stem = '/'.join(pieces[:2]), '/'.join(pieces[2:])
        folder = download(repo, [stem + '.json', stem + '.safetensors'])
        pipeline['args']['models'][key] = str(folder / stem)
    else:
        pipeline['args']['models'][key] = str(trellis / value)
dino = download(pipeline['args']['image_cond_model']['args']['model_name'])
pipeline['args']['image_cond_model']['args']['model_name'] = str(dino)
# Use the external SAM3 mask for both backends; do not load another segmenter.
pipeline['args']['rembg_model'] = {'name': 'ProvidedMask', 'args': {}}
prepared = target / 'prepared-trellis'
prepared.mkdir(exist_ok=True)
(prepared / 'pipeline.json').write_text(json.dumps(pipeline, indent=2))
(target / 'models.json').write_text(json.dumps({'flux2': str(flux), 'flux2_variant': sys.argv[2],
    'trellis2': str(prepared), 'model_revisions': revisions}, indent=2))
'''


if __name__ == '__main__':
    main()
