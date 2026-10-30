"""Managed Linux installation with separate model runtimes."""
import os
from pathlib import Path
import platform
import subprocess
import sys

from .common import read_json

ROOT = Path(__file__).resolve().parents[1]


def call(command, env):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def install(a):
    if platform.system() != 'Linux':
        raise ValueError('--setup requires Linux x86-64; other operating systems are untested')
    chosen = getattr(a, 'install_dir', None)
    scratch = Path(chosen or os.environ.get('SCRATCH', Path.home() / '.local/share/assetforge')).expanduser().resolve()
    base = (a.env_root or scratch / 'venv_for_assetforge').expanduser().resolve()
    if scratch not in base.parents:
        raise ValueError('The managed environment must be inside SCRATCH or --install-dir')
    scratch.mkdir(parents=True, exist_ok=True)
    if not a.environments_only and not a.from_run and not a.sam3d_pipeline:
        raise ValueError('Supply --sam3d-pipeline or --from-run; use --environments-only to install before downloading weights')
    env = os.environ.copy()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(SCRATCH=str(scratch), PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', MAX_JOBS=env.get('MAX_JOBS', '1'), ASSETFORGE_ENV_ROOT=str(base))
    if not (base / 'setup-complete.json').is_file():
        command = [sys.executable, ROOT / 'scripts/setup_level0.py', '--env-root', base]
        if a.resume:
            command.append('--resume')
        call(command, env)
    else:
        print('Reusing completed base environment:', base, flush=True)
    if getattr(a, 'with_blender', False):
        call([sys.executable, ROOT / 'scripts/setup_blender.py', '--env-root', base], env)
    if a.blender and not a.blender.resolve().is_file():
        raise ValueError('--blender must point to an existing executable file')
    if a.environments_only:
        print('Base environments ready. Obtain model access/weights, then rerun --setup with model paths.', flush=True)
        return
    command = [base / 'bin/python', '-B', '-m', 'assetforge.configuration', '--env-root', base, '--config', a.config.resolve()]
    for name in ('from_run', 'flux_model', 'sam3_model', 'sam_model', 'sam3d_pipeline', 'blender'):
        value = getattr(a, name)
        if value is not None:
            command.extend(['--' + name.replace('_', '-'), value.resolve()])
    call(command, env)
    tools = base.parent / 'assetforge-tools'
    deployment = read_json(a.config)
    pipeline = read_json(deployment['configs']['mesh'])['pipeline_config_path']
    call([base.with_name(base.name + '_sam3d') / 'bin/python', '-B', '-m', 'assetforge.offline',
          '--tools', tools, '--pipeline', pipeline], env)
    if a.with_extras:
        ready = tools / 'backend-comparison/ready.json'
        if ready.is_file():
            manifest = read_json(ready)
            if manifest['flux2_variant'] != a.flux2_variant:
                raise ValueError('Installed FLUX.2 variant differs; use a separate environment root for another variant')
            for prefix in manifest['prefixes'].values():
                if not (Path(prefix) / 'bin/python').is_file():
                    raise ValueError('Optional runtime missing; use the backend setup recipe with --resume')
            print('Reusing installed optional backends:', ready, flush=True)
        else:
            command = [sys.executable, ROOT / 'scripts/setup_backend_comparison.py', '--env-root', base,
                       '--flux2', a.flux2_variant]
            if a.resume:
                command.append('--resume')
            call(command, env)
    print('Asset-forge setup complete. GPU validation is performed by --test inside an allocation.', flush=True)
