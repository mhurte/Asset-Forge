"""Model access and download helper; executed in the small auth bootstrap env."""
import argparse
import json
from pathlib import Path

DEFAULTS = {
    'flux1': 'black-forest-labs/FLUX.1-dev',
    'sam3': 'facebook/sam3',
    'sam3d': 'facebook/sam-3d-objects',
    'sam': 'facebook/sam-vit-huge',
    'moge': 'Ruicheng/moge-vitl',
}
EXTRAS = ['black-forest-labs/FLUX.2-dev', 'microsoft/TRELLIS.2-4B',
          'microsoft/TRELLIS-image-large',
          'facebook/dinov3-vitl16-pretrain-lvd1689m']


def check_access(api, repos):
    checks = []
    for repo in repos:
        try:
            api.auth_check(repo_id=repo, repo_type='model')
            revision = api.model_info(repo).sha
            checks.append({'repo': repo, 'status': 'passed', 'revision': revision})
            print('[passed] access:', repo, flush=True)
        except Exception as exc:
            # Do not persist HTTP objects, headers or arbitrary exception messages.
            checks.append({'repo': repo, 'status': 'failed', 'error_type': type(exc).__name__})
            print('[failed] access:', repo, '(' + type(exc).__name__ + ')', flush=True)
            print('Check token permissions and account approval at https://huggingface.co/' + repo, flush=True)
    return checks


def prepare_pipeline(folder, models, yaml):
    candidates = sorted(folder.rglob('pipeline.yaml'))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one SAM3D pipeline.yaml; inspect the downloaded model layout')
    pipeline = candidates[0]
    def resolve(value, key=''):
        if isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, key) for v in value]
        if isinstance(value, str):
            if key == 'pretrained_model_name_or_path' and value == DEFAULTS['moge']:
                weight = models / 'moge/model.pt'
                if not weight.is_file():
                    raise FileNotFoundError('Downloaded MoGe model.pt is missing')
                return str(weight.resolve())
            if key.endswith(('_path', '_dir')) or key == 'pretrained_model_name_or_path':
                candidate = pipeline.parent / value.replace('${workspace_dir}', str(pipeline.parent.resolve()))
                if candidate.exists():
                    return str(candidate.resolve())
                if key.endswith('_path') or key == 'pretrained_model_name_or_path':
                    raise FileNotFoundError('Unresolved checkpoint/config reference before building: ' + value)
        return value
    data = resolve(yaml.safe_load(pipeline.read_text()))
    (models / 'prepared-pipeline.yaml').write_text(yaml.safe_dump(data))


def main():
    from huggingface_hub import HfApi, get_token, login, snapshot_download
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-dir', type=Path, required=True)
    p.add_argument('--with-extras', action='store_true')
    p.add_argument('--login', action='store_true')
    p.add_argument('--access-only', action='store_true')
    a = p.parse_args()
    if a.login:
        login(add_to_git_credential=False)
    token = get_token()
    if not token:
        print('No Hugging Face token found. Rerun install_test with --resume --login.', flush=True)
        raise SystemExit(2)
    root = a.install_dir.resolve()
    access_file = root / 'model-access.json'
    repos = list(DEFAULTS.values()) + (EXTRAS if a.with_extras else [])
    if a.access_only:
        checks = check_access(HfApi(token=token), repos)
        access_file.write_text(json.dumps({'checks': checks}, indent=2))
        if any(c['status'] != 'passed' for c in checks):
            raise SystemExit(2)
        return
    checks = json.loads(access_file.read_text())['checks']
    revisions = {c['repo']: c['revision'] for c in checks if c['status'] == 'passed'}
    models = root / 'models'
    models.mkdir(exist_ok=True)
    for name, repo in DEFAULTS.items():
        patterns = (['model_index.json', 'scheduler/*', 'text_encoder/*', 'text_encoder_2/*',
                     'tokenizer/*', 'tokenizer_2/*', 'transformer/*', 'vae/*', 'LICENSE*']
                    if name == 'flux1' else None)
        snapshot_download(repo_id=repo, revision=revisions[repo],
                          local_dir=str(models / name), max_workers=2,
                          allow_patterns=patterns, token=token)
    prepare_pipeline(models / 'sam3d', models, yaml)
    print('Default model downloads and local SAM3D pipeline prepared.', flush=True)


if __name__ == '__main__':
    main()
