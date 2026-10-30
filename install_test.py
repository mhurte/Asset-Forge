#!/usr/bin/env python3
"""Fresh online installation test. Run on Linux, before requesting GPU time."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-dir', type=Path, required=True)
    p.add_argument('--with-extras', action='store_true')
    p.add_argument('--login', action='store_true', help='Interactively log in to Hugging Face; never put a token in arguments')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--access-only', action='store_true', help='Check access without downloading model weights or building runtimes')
    a = p.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        p.error('The installation test requires Linux x86-64')
    root = a.install_dir.expanduser().resolve()
    marker = root / '.assetforge-install-test.json'
    if root.exists() and (not a.resume or not marker.is_file()):
        p.error('Choose a new installation directory, or use --resume for this test directory')
    if marker.is_file() and json.loads(marker.read_text()).get('with_extras') != a.with_extras:
        p.error('Resume with the same --with-extras selection')
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'with_extras': a.with_extras}))
    report = root / 'install-test.json'
    env = os.environ.copy()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    bootstrap = root / 'auth-bootstrap'
    python = bootstrap / 'bin/python'
    steps = []

    def stage(name, command, interactive=False):
        print('[running]', name, flush=True)
        steps.append({'step': name, 'status': 'running'})
        report.write_text(json.dumps({'steps': steps, 'gpu_validation': 'pending'}, indent=2))
        if interactive:
            # Login prompts are never redirected into the installation log.
            result = subprocess.run(list(map(str, command)), env=env, cwd=ROOT)
        else:
            log = root / (name + '.log')
            print('Log:', log, flush=True)
            with log.open('w') as output:
                result = subprocess.run(list(map(str, command)), env=env, cwd=ROOT,
                                        stdout=output, stderr=subprocess.STDOUT)
        steps[-1]['status'] = 'passed' if result.returncode == 0 else 'failed'
        report.write_text(json.dumps({'steps': steps, 'gpu_validation': 'pending'}, indent=2))
        print('[' + steps[-1]['status'] + ']', name, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    if not python.is_file():
        stage('bootstrap-venv', [sys.executable, '-m', 'venv', bootstrap])
    stage('bootstrap-packages', [python, '-m', 'pip', 'install', 'huggingface_hub==1.2.3', 'PyYAML==6.0.3'])
    probe = [python, '-B', '-m', 'assetforge.install_probe', '--install-dir', root]
    if a.with_extras:
        probe.append('--with-extras')
    access = [*probe, '--access-only']
    if a.login:
        access.append('--login')
    stage('model-access', access, interactive=True)
    if a.access_only:
        print('Access checks passed; no model weights or CUDA environments were installed.', flush=True)
        return
    stage('default-models', probe)
    models = root / 'models'
    command = [sys.executable, ROOT / 'assetgen.py', '--setup', '--resume', '--with-blender', '--install-dir', root,
               '--config', root / 'deployment.local.json', '--flux-model', models / 'flux1',
               '--sam3-model', models / 'sam3', '--sam-model', models / 'sam',
               '--sam3d-pipeline', models / 'prepared-pipeline.yaml']
    if a.with_extras:
        command.append('--with-extras')
    stage('runtime-setup', command)
    print('Online installation checks passed. GPU validation remains pending.', flush=True)
    print('Deployment:', root / 'deployment.local.json', flush=True)
    print('Report:', report, flush=True)


if __name__ == '__main__':
    main()
