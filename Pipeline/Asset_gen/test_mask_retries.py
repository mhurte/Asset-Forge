import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import image_to_mask as stage


class MaskRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'image.png'
        Image.new('RGB', (8, 8), 'white').save(self.source)
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'sam': {'enabled': False},
                                          'sam3': {'enabled': True, 'isolate_process': False}}))
        self.output = self.root / 'image_mask_sam3.png'
        self.device = patch.object(stage, '_resolve_device', return_value='cpu')
        self.device.start()

    def tearDown(self):
        self.device.stop()
        self.tmp.cleanup()

    def generate(self, **kwargs):
        return stage.generate_mask(self.source, 'object', config_path=self.config, backend='sam3', **kwargs)

    def test_two_failures_then_success_and_stop(self):
        with patch.object(stage, '_run_sam3', side_effect=[RuntimeError('first'), RuntimeError('second'), np.ones((8, 8))]) as run:
            result = self.generate()
        self.assertEqual(run.call_count, 3)
        self.assertEqual(len([f for f in result.failures if 'attempt' in f]), 2)
        self.assertTrue(result.mask_path.is_file())

    def test_first_success_is_not_repeated(self):
        with patch.object(stage, '_run_sam3', return_value=np.ones((8, 8))) as run:
            self.generate()
        self.assertEqual(run.call_count, 1)

    def test_exhaustion_cleans_stale_and_partial_outputs(self):
        Image.new('L', (8, 8), 255).save(self.output)
        def fail(*args):
            self.output.write_bytes(b'partial')
            raise RuntimeError('failure')
        with patch.object(stage, '_run_sam3', side_effect=fail) as run:
            with self.assertRaises(stage.MaskGenerationError):
                self.generate()
        self.assertEqual(run.call_count, 3)
        self.assertFalse(self.output.exists())

    def test_empty_and_wrong_dimensions_trigger_retries(self):
        for mask in (np.zeros((8, 8)), np.ones((4, 4))):
            with self.subTest(shape=mask.shape, empty=not mask.any()), \
                    patch.object(stage, '_run_sam3', return_value=mask) as run:
                with self.assertRaises(stage.MaskGenerationError):
                    self.generate()
                self.assertEqual(run.call_count, 3)
                self.assertFalse(self.output.exists())

    def test_retry_limit_can_be_disabled_or_rejected(self):
        with patch.object(stage, '_run_sam3', side_effect=RuntimeError('failure')) as run:
            with self.assertRaises(stage.MaskGenerationError):
                self.generate(max_attempts=1)
            self.assertEqual(run.call_count, 1)
        for limit in (0, -1, True, '3'):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.generate(max_attempts=limit)

    def test_isolated_backend_also_retries(self):
        data = json.loads(self.config.read_text())
        data['sam3']['isolate_process'] = True
        self.config.write_text(json.dumps(data))
        attempts = []
        def worker(*args):
            attempts.append(args)
            if len(attempts) == 1:
                raise RuntimeError('worker failed')
            Image.new('L', (8, 8), 255).save(args[2])
        with patch.object(stage, '_run_backend_isolated', side_effect=worker):
            self.generate()
        self.assertEqual(len(attempts), 2)


if __name__ == '__main__':
    unittest.main()
