import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch

from assetforge.install_probe import check_access, prepare_pipeline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fresh_install', ROOT / 'install_test.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallProbeTests(TestCase):
    def test_access_failure_is_reported_without_error_message_or_token(self):
        api = Mock()
        api.auth_check.side_effect = [RuntimeError('SECRET_MUST_NOT_BE_LOGGED'), None]
        api.model_info.return_value.sha = 'revision'
        with patch('builtins.print') as output:
            checks = check_access(api, ['gated/model', 'open/model'])
        self.assertEqual([c['status'] for c in checks], ['failed', 'passed'])
        self.assertNotIn('SECRET_MUST_NOT_BE_LOGGED', json.dumps(checks) + str(output.call_args_list))

    def test_auth_failure_stops_before_model_downloads_and_runtime_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fresh'
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                return Mock(returncode=2 if '--access-only' in command else 0)
            with patch.object(sys, 'argv', ['install_test.py', '--install-dir', str(root)]), \
                    patch.object(installer.platform, 'system', return_value='Linux'), \
                    patch.object(installer.platform, 'machine', return_value='x86_64'), \
                    patch.object(installer.subprocess, 'run', side_effect=run):
                with self.assertRaises(SystemExit) as result:
                    installer.main()
            self.assertEqual(result.exception.code, 2)
            self.assertFalse(any('assetgen.py' in str(c) for c in commands))
            report = json.loads((root / 'install-test.json').read_text())
            self.assertEqual(report['steps'][-1], {'step': 'model-access', 'status': 'failed'})
            self.assertEqual(report['gpu_validation'], 'pending')

    def test_full_install_test_always_requests_blender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'fresh'
            with patch.object(sys, 'argv', ['install_test.py', '--install-dir', str(root)]), \
                    patch.object(installer.platform, 'system', return_value='Linux'), \
                    patch.object(installer.platform, 'machine', return_value='x86_64'), \
                    patch.object(installer.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                installer.main()
            self.assertIn('--with-blender', run.call_args.args[0])

    def test_pipeline_resolves_workspace_and_moge_without_editing_download(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            folder = models / 'sam3d/checkpoints'
            folder.mkdir(parents=True)
            (folder / 'weights.ckpt').write_bytes(b'fixture')
            (models / 'moge').mkdir()
            (models / 'moge/model.pt').write_bytes(b'fixture')
            original = 'checkpoint_path: ${workspace_dir}/weights.ckpt\npretrained_model_name_or_path: Ruicheng/moge-vitl\n'
            pipeline = folder / 'pipeline.yaml'
            pipeline.write_text(original)
            prepare_pipeline(models / 'sam3d', models, yaml)
            prepared = yaml.safe_load((models / 'prepared-pipeline.yaml').read_text())
            self.assertEqual(prepared['checkpoint_path'], str((folder / 'weights.ckpt').resolve()))
            self.assertEqual(prepared['pretrained_model_name_or_path'], str((models / 'moge/model.pt').resolve()))
            self.assertEqual(pipeline.read_text(), original)
