"""Contract tests without downloading weights or importing CUDA dependencies."""
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experimental_backends import generate_flux2, generate_trellis2


class ExperimentalBackendTests(unittest.TestCase):
    def test_balanced_placement_uses_two_gpus_without_cpu_offload(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Mock()
            pipeline.hf_device_map = {'text_encoder': 0, 'transformer': 1, 'vae': 0}
            pipeline.return_value.images = [Mock()]
            cls = Mock()
            cls.from_pretrained.return_value = pipeline
            cuda = SimpleNamespace(device_count=lambda: 2, get_device_properties=lambda i: SimpleNamespace(total_memory=80 * 1024**3))
            torch = SimpleNamespace(bfloat16='bf16', Generator=Mock(), cuda=cuda)
            with patch.dict(sys.modules, {'torch': torch, 'diffusers': SimpleNamespace(Flux2Pipeline=cls, Flux2KleinPipeline=Mock())}):
                generate_flux2('test', '/model', Path(tmp) / 'image.png', device_strategy='balanced')
            self.assertEqual(cls.from_pretrained.call_args.kwargs['device_map'], 'balanced')
            self.assertEqual(set(cls.from_pretrained.call_args.kwargs['max_memory']), {0, 1})
            pipeline.enable_model_cpu_offload.assert_not_called()
            self.assertTrue((Path(tmp) / 'image.placement.json').is_file())

    def test_balanced_placement_fails_before_loading_if_only_one_gpu(self):
        cls = Mock()
        torch = SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(device_count=lambda: 1))
        with patch.dict(sys.modules, {'torch': torch, 'diffusers': SimpleNamespace(Flux2Pipeline=cls, Flux2KleinPipeline=Mock())}):
            with self.assertRaisesRegex(ValueError, 'two visible GPUs'):
                generate_flux2('test', '/model', '/unused.png', device_strategy='balanced')
        cls.from_pretrained.assert_not_called()

    def test_balanced_placement_rejects_cpu_spill(self):
        pipeline = Mock(hf_device_map={'transformer': 0, 'text_encoder': 'cpu', 'vae': 1})
        cls = Mock()
        cls.from_pretrained.return_value = pipeline
        torch = SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(device_count=lambda: 2,
            get_device_properties=lambda i: SimpleNamespace(total_memory=80 * 1024**3)))
        with patch.dict(sys.modules, {'torch': torch, 'diffusers': SimpleNamespace(Flux2Pipeline=cls, Flux2KleinPipeline=Mock())}):
            with self.assertRaisesRegex(RuntimeError, 'CPU/disk spill'):
                generate_flux2('test', '/model', '/unused.png', device_strategy='balanced')
        pipeline.assert_not_called()

    def test_flux2_variants_have_separate_sampling_settings(self):
        for variant, steps, guidance in [('dev', 50, 2.5), ('klein', 4, 1.0)]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                pipeline = Mock()
                pipeline.return_value.images = [Mock()]
                dev, klein = Mock(), Mock()
                (dev if variant == 'dev' else klein).from_pretrained.return_value = pipeline
                torch = SimpleNamespace(bfloat16='bf16', Generator=Mock())
                with patch.dict(sys.modules, {'torch': torch, 'diffusers': SimpleNamespace(
                        Flux2Pipeline=dev, Flux2KleinPipeline=klein)}):
                    generate_flux2('test', '/local/model', Path(tmp) / 'image.png', variant=variant)
                self.assertEqual(pipeline.call_args.kwargs['num_inference_steps'], steps)
                self.assertEqual(pipeline.call_args.kwargs['guidance_scale'], guidance)
                self.assertTrue((dev if variant == 'dev' else klein).from_pretrained.call_args.kwargs['local_files_only'])

    def test_trellis_preserves_supplied_mask_and_exports_standard_glb(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (16, 16), 'yellow').save(root / 'image.png')
            mask = Image.new('L', (16, 16))
            mask.paste(255, (2, 2, 14, 14))
            mask.save(root / 'mask.png')
            mesh = SimpleNamespace(**{k: k for k in ['vertices', 'faces', 'attrs', 'coords', 'layout', 'voxel_size']}, simplify=Mock())
            pipeline = Mock()
            pipeline.run.return_value = [mesh]
            cls = Mock()
            cls.from_pretrained.return_value = pipeline
            glb = Mock()
            glb.export.side_effect = lambda path, **kwargs: Path(path).write_bytes(b'export fixture' * 4)
            export = Mock(return_value=glb)
            rembg = SimpleNamespace()
            modules = {'trellis2': SimpleNamespace(), 'trellis2.pipelines': SimpleNamespace(
                Trellis2ImageTo3DPipeline=cls, rembg=rembg), 'trellis2.pipelines.rembg': rembg,
                'o_voxel': SimpleNamespace(postprocess=SimpleNamespace(to_glb=export))}
            with patch.dict(sys.modules, modules):
                generate_trellis2(root / 'image.png', root / 'mask.png', root / 'mesh.glb',
                                 repository_path=root, model_path=root, seed=123)
            self.assertEqual(pipeline.run.call_args.args[0].getchannel('A').tobytes(), mask.tobytes())
            self.assertEqual(pipeline.run.call_args.kwargs['seed'], 123)
            self.assertFalse(glb.export.call_args.kwargs['extension_webp'])
            self.assertEqual(export.call_args.kwargs['attr_layout'], mesh.layout)


if __name__ == '__main__':
    unittest.main()
