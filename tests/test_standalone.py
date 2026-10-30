import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from assetforge.cli import CUP, main, parser
from assetforge.common import write_json
from assetforge.runtime import Run
from assetforge.worker import masks

ROOT = Path(__file__).resolve().parents[1]


def glb_fixture(path):
    doc = {'asset': {'version': '2.0'}, 'meshes': [{'primitives': [{'attributes': {'POSITION': 0}}]}],
           'buffers': [{'byteLength': 36}], 'bufferViews': [{'buffer': 0, 'byteLength': 36}],
           'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3', 'min': [0,0,0], 'max': [1,1,0]}],
           'nodes': [{'mesh': 0}], 'scenes': [{'nodes': [0]}], 'scene': 0}
    encoded = json.dumps(doc).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary = struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0)
    path.write_bytes(struct.pack('<III', 0x46546c67, 2, 28 + len(encoded) + len(binary)) +
                     struct.pack('<I4s', len(encoded), b'JSON') + encoded + struct.pack('<I4s', len(binary), b'BIN\0') + binary)


class StandaloneTests(unittest.TestCase):
    def test_defaults_and_optional_test_prompt(self):
        a = parser().parse_args(['--test'])
        self.assertEqual(a.test, CUP)
        self.assertEqual((a.image_backend, a.mask_backend, a.mesh_backend), ('flux', 'sam3', 'sam3d'))
        self.assertEqual(parser().parse_args(['--test', 'a man holding a mug']).test, 'a man holding a mug')

    def exercise(self, options, fail=None, blender_available=True):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / 'env'
            tools = root / 'tools'
            (tools / 'backend-comparison').mkdir(parents=True)
            model = root / 'model'
            model.mkdir()
            mask_config = {name: {'checkpoint_path': str(model), 'enabled': True} for name in ('sam', 'sam3')}
            cfgs = {}
            for name, value in [('mask', mask_config), ('image', {}), ('mesh', {})]:
                cfgs[name] = str(root / (name + '.json'))
                write_json(cfgs[name], value)
            blender = root / 'fake-blender'
            blender.write_bytes(b'fixture')
            write_json(root / 'deployment.json', {'env_root': str(prefix), 'tools': str(tools), 'blender': str(blender) if blender_available else None, 'configs': cfgs})
            write_json(tools / 'backend-comparison/ready.json', {'flux2': '/model', 'flux2_variant': 'dev',
                'prefixes': {'flux2': str(prefix), 'trellis2': str(prefix)}, 'sources': {'trellis2': '/source'}, 'trellis2': '/model'})
            requests = []
            def fake(run, name, command, timeout, env, reserve=0):
                run.report['steps'].append({'name': name, 'status': 'failed' if name == fail else 'passed'})
                if name == fail:
                    return False
                if name.endswith('/render'):
                    folder = Path(command[command.index('--output') + 1])
                    for view in ('side', 'high', 'reverse'):
                        (folder / ('studio-agx-' + view + '.png')).write_bytes(b'png')
                else:
                    r = json.loads(Path(command[-1]).read_text())
                    requests.append((name, r))
                    if name.endswith('/image'):
                        Image.new('RGB', (16, 16), 'yellow').save(r.get('image', r.get('output_path')))
                    elif name.endswith('/mask'):
                        output = Path(r['directory']) / 'mask.png'
                        Image.new('L', (16, 16), 128).save(output)
                        write_json(r['result'], {'mask': str(output)})
                    else:
                        glb = Path(r['ply']).with_suffix('.glb') if 'ply' in r else Path(r['output_path'])
                        glb_fixture(glb)
                        if 'ply' in r:
                            Path(r['ply']).write_bytes(b'ply fixture')
                run.save()
                return True
            with patch.object(Run, 'execute', fake):
                with self.assertRaises(SystemExit) as stop:
                    main([*options, '--config', str(root / 'deployment.json'), '--output', str(root / 'out')])
            report = json.loads((root / 'out/report.json').read_text())
            self.assertEqual(stop.exception.code, 2 if fail else 0)
            copied = json.loads((root / 'out/configs/mask.json').read_text())
            self.assertTrue(copied[report['subject']['mask_backend']]['enabled'])
            other = 'sam' if report['subject']['mask_backend'] == 'sam3' else 'sam3'
            self.assertFalse(copied[other]['enabled'])
            return report, requests

    def test_default_generation_only_uses_default_models(self):
        report, requests = self.exercise(['--prompt', 'a mug', '--mask-prompts', 'mug'])
        self.assertEqual([r['id'] for r in report['assets']], ['flux-sam3d'])
        self.assertIn('ply', report['assets'][0])

    def test_full_comparison_shares_images_and_masks(self):
        report, requests = self.exercise(['--test'])
        self.assertEqual(len(report['assets']), 4)
        for image in ('flux', 'flux2'):
            pair = [r for r in report['assets'] if r['image_backend'] == image]
            self.assertEqual(pair[0]['image_sha256'], pair[1]['image_sha256'])
            self.assertEqual(pair[0]['mask_sha256'], pair[1]['mask_sha256'])

    def test_comparison_without_blender_previews(self):
        report, requests = self.exercise(['--test', '--no-render'], blender_available=False)
        self.assertEqual(len(report['assets']), 4)
        self.assertFalse(any(s['name'].endswith('/render') for s in report['steps']))

    def test_sam_is_an_explicit_comparison_option(self):
        report, requests = self.exercise(['--test', '--mask-backend', 'sam'])
        self.assertTrue(all(r['mask_backend'] == 'sam' for r in report['assets']))

    def test_one_backend_failure_retains_other_comparison_cells(self):
        report, requests = self.exercise(['--test'], fail='flux-trellis2/mesh')
        self.assertEqual(sum(r['status'] == 'passed' for r in report['assets']), 3)

    def test_only_optional_models_do_not_claim_ply(self):
        report, requests = self.exercise(['--prompt', 'a mug', '--mask-prompts', 'mug', '--image-backend', 'flux2', '--mesh-backend', 'trellis2'])
        self.assertEqual(report['assets'][0]['id'], 'flux2-trellis2')
        self.assertNotIn('ply', report['assets'][0])


class MaskCombinationTests(unittest.TestCase):
    def test_classic_sam_runs_automatic_proposals_once_for_multiple_objects(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def automatic(image, prompt, **kwargs):
                self.assertIsNone(prompt)
                path = Path(kwargs['output_dir']) / 'mask.png'
                result = Image.new('L', (8, 8))
                result.putpixel((1, 1), 255)
                result.putpixel((6, 6), 255)
                result.save(path)
                return SimpleNamespace(mask_paths={'sam': path})
            generator = Mock(side_effect=automatic)
            result = masks({'backend': 'sam', 'mask_prompts': ['player', 'ball'], 'directory': str(root),
                            'image': 'input', 'config': 'config'}, generator)
            generator.assert_called_once()
            with Image.open(result['mask']) as mask:
                self.assertEqual((mask.getpixel((1,1)), mask.getpixel((6,6))), (255,255))

    def test_sam3_empty_second_part_cannot_be_reported_as_complete_union(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def generator(image, prompt, **kwargs):
                path = Path(kwargs['output_dir']) / 'mask.png'
                Image.new('L', (8,8), 255 if prompt == 'man' else 0).save(path)
                return SimpleNamespace(mask_paths={'sam3': path})
            with self.assertRaisesRegex(ValueError, 'Empty'):
                masks({'backend': 'sam3', 'mask_prompts': ['man', 'mug'], 'directory': str(root),
                       'image': 'input', 'config': 'config'}, generator)
            self.assertFalse((root / 'foreground-mask.png').exists())

    def test_sam3_components_are_required_and_unioned(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def generator(image, prompt, **kwargs):
                path = Path(kwargs['output_dir']) / 'mask.png'
                result = Image.new('L', (8, 8))
                result.putpixel((1, 1) if prompt == 'man' else (6, 6), 255)
                result.save(path)
                return SimpleNamespace(mask_paths={'sam3': path})
            result = masks({'backend': 'sam3', 'mask_prompts': ['man', 'mug'], 'directory': str(root), 'image': 'input', 'config': 'config'}, generator)
            mask = Image.open(result['mask'])
            self.assertEqual((mask.getpixel((1,1)), mask.getpixel((6,6))), (255,255))

    def test_classic_sam_combines_disjoint_proposals_by_default(self):
        import numpy as np
        spec = importlib.util.spec_from_file_location('mask_impl', ROOT / 'Pipeline/Asset_gen/image_to_mask.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        one = np.zeros((8, 8), dtype=bool)
        two = one.copy()
        one[1:3, 1:3], two[6, 6] = True, True
        generator = Mock(return_value={'masks': [one, two], 'scores': [1, 1]})
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(pipeline=Mock(return_value=generator))}), \
                patch.object(module, '_require_path', return_value=Path('/model')), \
                patch.object(module, '_load_rgb_image', return_value=np.zeros((8,8,3), dtype=np.uint8)):
            union = module._run_sam(Path('image'), {}, Path('/'), 'cpu')
            largest = module._run_sam(Path('image'), {'mask_combination': 'largest'}, Path('/'), 'cpu')
        self.assertEqual(int(union.sum()), 5)
        self.assertEqual(int(largest.sum()), 4)


if __name__ == '__main__':
    unittest.main()
