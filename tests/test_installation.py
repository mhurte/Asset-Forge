import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assetforge.common import read_json, write_json
from assetforge.configuration import prepare
from assetforge.installation import install


class InstallationTests(unittest.TestCase):
    def test_custom_install_directory_without_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'chosen-storage'
            a = self.options(root)
            a.install_dir = root
            a.env_root = None
            a.environments_only = True
            with patch('assetforge.installation.platform.system', return_value='Linux'), \
                    patch.dict(os.environ, {}, clear=True), patch('assetforge.installation.call') as call:
                install(a)
            command, environment = call.call_args.args
            self.assertEqual(command[command.index('--env-root') + 1], root.resolve() / 'venv_for_assetforge')
            self.assertEqual(environment['SCRATCH'], str(root.resolve()))
            self.assertEqual(environment['ASSETFORGE_ENV_ROOT'], str(root.resolve() / 'venv_for_assetforge'))

    def test_install_default_without_cluster_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.options(Path(tmp))
            a.env_root = None
            a.environments_only = True
            with patch('assetforge.installation.platform.system', return_value='Linux'), \
                    patch('assetforge.installation.Path.home', return_value=Path(tmp)), \
                    patch.dict(os.environ, {}, clear=True), patch('assetforge.installation.call') as call:
                install(a)
            self.assertEqual(call.call_args.args[1]['SCRATCH'], str((Path(tmp) / '.local/share/assetforge').resolve()))

    def options(self, root):
        return SimpleNamespace(env_root=root / 'env', config=root / 'assetforge.local.json',
            from_run=root / 'prior', sam3d_pipeline=None, environments_only=False,
            resume=True, with_extras=False, flux2_variant='dev', flux_model=None,
            sam3_model=None, sam_model=None, blender=None)

    def test_existing_install_reuses_environments_and_skips_optional_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.options(root)
            a.env_root.mkdir()
            (a.env_root / 'setup-complete.json').write_text('{}')
            write_json(a.config, {'configs': {'mesh': str(root / 'mesh.json')}})
            write_json(root / 'mesh.json', {'pipeline_config_path': '/existing/pipeline.yaml'})
            with patch('assetforge.installation.platform.system', return_value='Linux'), \
                    patch.dict(os.environ, {'SCRATCH': str(root)}), patch('assetforge.installation.call') as call:
                install(a)
            calls = [' '.join(map(str, c.args[0])) for c in call.call_args_list]
            self.assertEqual(len(calls), 2)
            self.assertIn('assetforge.configuration', calls[0])
            self.assertIn('assetforge.offline', calls[1])
            self.assertTrue(all('setup_backend_comparison' not in c and 'setup_level0.py' not in c for c in calls))

    def test_opt_in_blender_is_installed_even_when_base_is_already_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.options(root)
            a.with_blender = True
            a.environments_only = True
            a.env_root.mkdir()
            (a.env_root / 'setup-complete.json').write_text('{}')
            with patch('assetforge.installation.platform.system', return_value='Linux'), \
                    patch.dict(os.environ, {'SCRATCH': str(root)}), patch('assetforge.installation.call') as call:
                install(a)
            self.assertEqual(len(call.call_args_list), 1)
            self.assertTrue(str(call.call_args.args[0][1]).endswith('setup_blender.py'))

    def test_environment_outside_scratch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.options(Path(tmp))
            with patch('assetforge.installation.platform.system', return_value='Linux'), \
                    patch.dict(os.environ, {'SCRATCH': str(Path(tmp) / 'scratch')}), \
                    patch('assetforge.installation.call') as call:
                with self.assertRaisesRegex(ValueError, 'inside SCRATCH'):
                    install(a)
                call.assert_not_called()

    def test_adoption_copies_configs_without_changing_source(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.options(root)
            source = a.from_run / 'configs'
            source.mkdir(parents=True)
            model = root / 'model'
            model.mkdir()
            pipeline = source / 'pipeline.yaml'
            pipeline.write_text('checkpoint_path: weights.pt\nrendering_engine: original\n')
            (source / 'weights.pt').write_bytes(b'fixture')
            original = pipeline.read_text()
            write_json(source / 'image.json', {'model_path': str(model), 'inference_options': {}})
            write_json(source / 'mask.json', {name: {'checkpoint_path': str(model)} for name in ('sam', 'sam3')})
            write_json(source / 'mesh.json', {'pipeline_config_path': str(pipeline)})
            deployment = prepare(a)
            mesh = read_json(deployment['configs']['mesh'])
            copied = yaml.safe_load(Path(mesh['pipeline_config_path']).read_text())
            self.assertEqual(copied['checkpoint_path'], str((source / 'weights.pt').resolve()))
            self.assertEqual(copied['rendering_engine'], 'pytorch3d')
            self.assertEqual(pipeline.read_text(), original)
            self.assertTrue(read_json(deployment['configs']['mesh'])['export_glb'])
            self.assertEqual(read_json(deployment['configs']['mask'])['sam']['mask_combination'], 'union')


if __name__ == '__main__':
    unittest.main()
