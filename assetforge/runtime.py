"""Standalone failure-tolerant orchestration and comparison gallery."""
import html
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time

from .common import file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'assetforge/worker.py'
ADAPTER = ROOT / 'Pipeline/Asset_gen/experimental_backends.py'


def environment(prefix):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OPENCV_IO_ENABLE_OPENEXR='1', CONDA_PREFIX=str(prefix), CUDA_HOME=str(prefix), SPARSE_BACKEND='spconv')
    env['PATH'] = str(prefix / 'bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, [prefix / 'lib', prefix / 'targets/x86_64-linux/lib'])) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def validate_glb(path):
    with path.open('rb') as source:
        header = source.read(20)
        if len(header) != 20:
            raise ValueError('Truncated GLB')
        magic, version, length, size, chunk = struct.unpack('<5I', header)
        if magic != 0x46546c67 or version != 2 or length != path.stat().st_size or chunk != 0x4e4f534a or size > length - 20:
            raise ValueError('Invalid GLB header')
        doc = json.loads(source.read(size))
    if not any(p.get('mode', 4) in (4, 5, 6) for m in doc.get('meshes', []) for p in m.get('primitives', [])):
        raise ValueError('Output has no triangle mesh')


class Run:
    def __init__(self, output, seconds, subject):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.deadline = time.monotonic() + seconds
        self.report = {'status': 'running', 'subject': subject, 'steps': [], 'assets': []}
        self.save()
        print('Asset-forge output:', self.output, flush=True)

    def save(self):
        write_json(self.output / 'report.json', self.report)
        rows = ['# Asset-forge\n\nStatus: ' + self.report['status'], '| Step | Status |', '| --- | --- |']
        rows += ['| ' + s['name'] + ' | ' + s['status'] + ' |' for s in self.report['steps']]
        (self.output / 'report.md').write_text('\n'.join(rows) + '\n')
        page = ['<!doctype html><meta charset="utf-8"><title>Asset-forge</title>',
                '<style>body{font:16px system-ui;background:#ddd;margin:24px}img{width:30%;max-width:1024px}pre{white-space:pre-wrap}</style>',
                '<h1>Asset-forge</h1><p>' + html.escape(self.report['subject']['prompt'] or 'Existing image') + '</p>',
                '<p>Each reconstruction pair shares an image and mask. Camera directions and lighting are shared; asset orientation may differ.</p>',
                '<pre>' + html.escape((self.output / 'report.md').read_text()) + '</pre>']
        for path in sorted((self.output / 'images').glob('*/image.png')):
            rel = path.relative_to(self.output).as_posix()
            page += ['<h2>' + path.parent.name + ' source</h2>', f'<a href="{rel}"><img src="{rel}"></a>']
        for asset in self.report['assets']:
            page.append('<h2>' + asset['id'] + '</h2><p>' + html.escape(asset['status']) + '</p>')
            for kind in ('glb', 'ply'):
                if asset.get(kind):
                    relative = html.escape(asset[kind], quote=True)
                    page.append(f'<a href="{relative}">Open {kind.upper()}</a> ')
            for view in ('side', 'high', 'reverse', 'upper-side', 'upper-reverse'):
                rel = 'renders/' + asset['id'] + '/studio-agx-' + view + '.png'
                if (self.output / rel).is_file():
                    page.append(f'<a href="{rel}"><img src="{rel}"></a>')
        (self.output / 'index.html').write_text(''.join(page), encoding='utf-8')

    def execute(self, name, command, timeout, env, reserve=0):
        remaining = min(timeout, self.deadline - time.monotonic() - reserve)
        row = {'name': name, 'status': 'running', 'command': list(map(str, command))}
        self.report['steps'].append(row)
        if remaining <= 0:
            row['status'] = 'skipped: budget exhausted'
            self.save()
            return False
        log = self.output / 'logs' / (name.replace('/', '_') + '.log')
        log.parent.mkdir(exist_ok=True)
        row['log'] = str(log.relative_to(self.output))
        self.save()
        print('[running]', name, 'log:', log, flush=True)
        start = time.monotonic()
        try:
            with log.open('w') as stream:
                process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=os.name == 'posix')
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                    raise TimeoutError('Stage timeout')
            row['status'] = 'passed' if code == 0 else 'failed: exit ' + str(code)
        except (OSError, TimeoutError) as exc:
            row['status'] = 'failed: ' + str(exc)
        row['elapsed_seconds'] = round(time.monotonic() - start, 2)
        self.save()
        print('[' + row['status'] + ']', name, flush=True)
        return row['status'] == 'passed'


def generate(a):
    if not a.config.is_file():
        raise ValueError('Deployment config missing; run --setup first or pass --config')
    deployment = read_json(a.config)
    want_render = a.render or (a.test is not None and not getattr(a, 'no_render', False))
    if want_render and (not deployment.get('blender') or not Path(deployment['blender']).is_file()):
        raise ValueError('Previews require Blender: rerun setup with --with-blender or --blender /path/to/blender; use --test --no-render to compare without previews')
    base, tools = Path(deployment['env_root']), Path(deployment['tools'])
    images = ['flux', 'flux2'] if a.test is not None else ['input' if a.image else a.image_backend]
    meshes = ['sam3d', 'trellis2'] if a.test is not None else [a.mesh_backend]
    extras = None
    if 'flux2' in images or 'trellis2' in meshes:
        path = tools / 'backend-comparison/ready.json'
        if not path.is_file():
            raise ValueError('Optional models are not installed; run --setup --with-extras --resume')
        extras = read_json(path)
    mask_settings = read_json(deployment['configs']['mask'])
    if not a.mask and not Path(mask_settings[a.mask_backend]['checkpoint_path']).is_dir():
        raise ValueError('Selected mask model is missing; configure its local checkpoint directory with --setup')
    subject = {'label': a.label, 'prompt': a.test if a.test is not None else a.prompt,
               'mask_prompts': a.mask_prompts, 'mask_backend': a.mask_backend, 'seed': a.seed}
    run = Run(a.output, a.budget, subject)
    run.report['config'] = deployment
    run.report['code_sha256'] = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / 'Pipeline/Asset_gen').glob('*.py'))}
    run.report['optional_models'] = extras
    run.report['orchestration_sha256'] = {p.name: file_digest(p) for p in sorted((ROOT / 'assetforge').glob('*.py'))}
    run.report['flux2_device_strategy'] = a.flux2_device_strategy
    configs = run.output / 'configs'
    configs.mkdir()
    for name, source in deployment['configs'].items():
        shutil.copyfile(source, configs / (name + '.json'))
    mesh_config = read_json(configs / 'mesh.json')
    if mesh_config.get('pipeline_config_path'):
        shutil.copyfile(mesh_config['pipeline_config_path'], configs / 'pipeline.yaml')
        mesh_config['pipeline_config_path'] = str(configs / 'pipeline.yaml')
        write_json(configs / 'mesh.json', mesh_config)
    mask_config = read_json(configs / 'mask.json')
    for name in ('sam', 'sam3'):
        mask_config[name].update(enabled=name == a.mask_backend, isolate_process=False)
    write_json(configs / 'mask.json', mask_config)
    run.save()
    renders = []
    for image_backend in images:
        folder = run.output / 'images' / image_backend
        folder.mkdir(parents=True)
        image = folder / 'image.png'
        common = {'image': str(image), 'seed': a.seed, 'directory': str(folder), 'mask_prompts': a.mask_prompts}
        def stage(name, request, prefix, timeout=600, reserve=300):
            path = folder / (name + '-request.json')
            write_json(path, request)
            return run.execute(image_backend + '/' + name, [prefix / 'bin/python', '-B', WORKER, name, path],
                               timeout, environment(prefix), reserve)
        if image_backend == 'input':
            shutil.copyfile(a.image.resolve(), image)
            ok = True
        elif image_backend == 'flux':
            ok = stage('image', {**common, 'prompt': subject['prompt'], 'config': str(configs / 'image.json'),
                                'result': str(folder / 'image-result.json')}, base)
        else:
            request = folder / 'flux2-request.json'
            write_json(request, {'prompt': subject['prompt'], 'model_path': extras['flux2'], 'variant': extras['flux2_variant'],
                                 'output_path': str(image), 'seed': a.seed, 'resolution': 1024, 'device_strategy': a.flux2_device_strategy})
            prefix = Path(extras['prefixes']['flux2'])
            ok = run.execute('flux2/image', [prefix / 'bin/python', '-B', ADAPTER, 'flux2', request], 600, environment(prefix), 300)
        mask = a.mask.resolve() if a.mask else None
        if ok and not mask:
            ok = stage('mask', {**common, 'backend': a.mask_backend, 'config': str(configs / 'mask.json'),
                               'result': str(folder / 'mask-result.json')}, base, 540)
            if ok:
                mask = Path(read_json(folder / 'mask-result.json')['mask'])
        for mesh_backend in meshes:
            row = {'id': image_backend + '-' + mesh_backend, 'image_backend': image_backend, 'mesh_backend': mesh_backend,
                   'mask_backend': 'provided' if a.mask else a.mask_backend, 'status': 'pending'}
            run.report['assets'].append(row)
            if not ok:
                row['status'] = 'skipped: image or mask failed'
                run.save()
                continue
            row.update(image_sha256=file_digest(image), mask_sha256=file_digest(mask))
            directory = run.output / 'assets' / row['id']
            directory.mkdir(parents=True)
            glb = directory / (a.label + '.glb')
            if mesh_backend == 'sam3d':
                prefix = base.with_name(base.name + '_sam3d')
                request = {**common, 'mask': str(mask), 'ply': str(glb.with_suffix('.ply')), 'tools': str(tools),
                           'config': str(configs / 'mesh.json'), 'result': str(directory / 'result.json')}
                path = directory / 'request.json'
                write_json(path, request)
                command = [prefix / 'bin/python', '-B', WORKER, 'mesh', path]
            else:
                prefix = Path(extras['prefixes']['trellis2'])
                path = directory / 'request.json'
                write_json(path, {'image_path': str(image), 'mask_path': str(mask), 'output_path': str(glb),
                                 'repository_path': extras['sources']['trellis2'], 'model_path': extras['trellis2'], 'seed': a.seed})
                command = [prefix / 'bin/python', '-B', ADAPTER, 'trellis2', path]
            ok_mesh = run.execute(row['id'] + '/mesh', command, 600, environment(prefix), 300)
            try:
                if not ok_mesh:
                    raise ValueError('Mesh stage failed')
                validate_glb(glb)
                row.update(status='passed', glb=str(glb.relative_to(run.output)), glb_sha256=file_digest(glb))
                if mesh_backend == 'sam3d':
                    ply = glb.with_suffix('.ply')
                    if not ply.is_file() or not ply.stat().st_size:
                        raise ValueError('SAM3D Gaussian PLY missing')
                    row.update(ply=str(ply.relative_to(run.output)), ply_sha256=file_digest(ply))
                renders.append((row, glb))
            except (OSError, ValueError) as exc:
                row['status'] = 'failed: ' + str(exc)
            run.save()
    if want_render:
        for row, glb in renders:
            folder = run.output / 'renders' / row['id']
            folder.mkdir(parents=True)
            ok = run.execute(row['id'] + '/render', [deployment['blender'], '--background', '--factory-startup', '--python-exit-code', '1',
                '--python', ROOT / 'assetforge/blender_asset_comparison.py', '--', '--asset', glb, '--label', a.label,
                '--output', folder, '--device', 'GPU', '--resolution', '1024' if a.label == 'MAN_HOLDING_CUP' else '768',
                '--samples', '96', '--setups', 'studio-agx'], 300, environment(base))
            expected = 5 if a.label == 'MAN_HOLDING_CUP' else 3
            if not ok or len(list(folder.glob('studio-agx-*.png'))) != expected:
                row['status'] = 'failed: render incomplete; mesh retained'
            run.save()
    passed = len(run.report['assets']) == len(images) * len(meshes) and all(r['status'] == 'passed' for r in run.report['assets'])
    run.report['status'] = 'passed' if passed else 'degraded'
    run.save()
    print(run.report['status'] + ': ' + str(run.output / 'index.html'), flush=True)
    return 0 if passed else 2
