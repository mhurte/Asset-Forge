import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from assetforge.configuration import prepare

ROOT = Path(__file__).resolve().parents[1]


class PublicTemplateTests(unittest.TestCase):
    def test_fresh_configuration_resolves_supplied_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'models'
            model.mkdir()
            pipeline = model / 'pipeline.yaml'
            pipeline.write_text('checkpoint_path: weights.pt\n')
            (model / 'weights.pt').write_bytes(b'fixture')
            args = SimpleNamespace(env_root=root / 'runtime', config=root / 'deployment.local.json',
                from_run=None, flux_model=model, sam3_model=model, sam_model=model,
                sam3d_pipeline=pipeline, blender=None)
            deployment = prepare(args)
            self.assertIn('assetforge-tools', deployment['tools'])
            for config in deployment['configs'].values():
                text = Path(config).read_text()
                self.assertNotIn('/lustre/', text)
                self.assertNotIn('/path/to/', text)
            image = json.loads(Path(deployment['configs']['image']).read_text())
            self.assertEqual(image['model_path'], str(model.resolve()))


if __name__ == '__main__':
    unittest.main()
