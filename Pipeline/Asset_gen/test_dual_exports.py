"""CPU-only checks for paired exports; run with python -m unittest test_dual_exports."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import asset_generation_pipeline as pipeline
import image_mask_to_3d as stage


class DualExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.destination = self.root / "assets" / "chair.ply"
        self.output = {
            "gs": SimpleNamespace(save_ply=lambda p: Path(p).write_bytes(b"splat")),
            "glb": SimpleNamespace(export=lambda p: Path(p).write_bytes(b"mesh")),
        }

    def test_default_saves_both(self):
        mesh = stage.export_assets(self.output, self.destination)
        self.assertEqual(mesh, self.destination.with_suffix(".glb"))
        self.assertEqual(self.destination.read_bytes(), b"splat")
        self.assertEqual(mesh.read_bytes(), b"mesh")

    def test_missing_mesh_requires_explicit_ply_only(self):
        del self.output["glb"]
        with self.assertRaisesRegex(stage.AssetGenerationError, "decode_formats"):
            stage.export_assets(self.output, self.destination)
        self.assertIsNone(stage.export_assets(self.output, self.destination, export_glb=False))
        self.assertTrue(self.destination.is_file())

    def test_failed_or_empty_mesh_is_not_success(self):
        for exporter in (Mock(side_effect=RuntimeError("export error")), lambda p: Path(p).touch()):
            with self.subTest(exporter=exporter):
                self.output["glb"] = SimpleNamespace(export=exporter)
                with self.assertRaises(stage.AssetGenerationError):
                    stage.export_assets(self.output, self.destination)

    def test_worker_handoff_moves_both_files(self):
        def worker(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            stage.export_assets(self.output, output)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(stage.subprocess, "run", side_effect=worker):
            stage._run_isolated(Path(sys.executable), self.root / "image.png",
                                self.root / "mask.png", self.root / "config.json",
                                self.destination, 42, 60)
        self.assertEqual(self.destination.read_bytes(), b"splat")
        self.assertEqual(self.destination.with_suffix(".glb").read_bytes(), b"mesh")
        self.assertEqual(len(list(self.destination.parent.iterdir())), 2)

    def test_worker_missing_mesh_is_rejected_and_cleaned(self):
        def worker(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"splat")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch.object(stage.subprocess, "run", side_effect=worker):
            with self.assertRaisesRegex(stage.AssetGenerationError, "all requested"):
                stage._run_isolated(Path(sys.executable), self.root / "image.png",
                                    self.root / "mask.png", self.root / "config.json",
                                    self.destination, 42, 60)
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_generation_calls_inference_once_and_returns_both(self):
        image, mask = self.root / "image.png", self.root / "mask.png"
        image.touch()
        mask.touch()
        inference_module, checkpoint = self.root / "inference.py", self.root / "pipeline.yaml"
        inference_module.touch()
        checkpoint.touch()
        config = self.root / "config.json"
        config.write_text(json.dumps({"repository_path": str(self.root),
                                     "inference_module_path": str(inference_module),
                                     "pipeline_config_path": str(checkpoint)}))
        inference = Mock(return_value=self.output)
        with patch.object(stage, "_load_inputs", return_value=("image", "mask")), \
                patch.object(stage, "_create_inference", return_value=inference):
            result = stage.generate_3d_asset(image, mask, config_path=config,
                                             output_path=self.destination)
        inference.assert_called_once_with("image", "mask", seed=42)
        self.assertEqual(result.asset_path, self.destination)
        self.assertEqual(result.mesh_path, self.destination.with_suffix(".glb"))

    def test_pipeline_exposes_paired_paths(self):
        image, mask_config, asset_config = (self.root / n for n in
                                            ("image.png", "mask.json", "asset.json"))
        for path in (image, mask_config, asset_config):
            path.touch()
        config = self.root / "pipeline.json"
        config.write_text(json.dumps({"image_to_mask_config": str(mask_config),
                                     "image_mask_to_3d_config": str(asset_config)}))
        mask = self.root / "image_mask_sam3.png"
        mesh = self.destination.with_suffix(".glb")
        modules = {
            "image_to_mask": SimpleNamespace(generate_mask=Mock(return_value=
                SimpleNamespace(mask_paths={"sam3": mask}, failures=()))),
            "image_mask_to_3d": SimpleNamespace(generate_3d_asset=Mock(return_value=
                SimpleNamespace(asset_path=self.destination, mesh_path=mesh))),
        }
        with patch.dict(sys.modules, modules):
            result = pipeline.generate_asset_pipeline(image_path=image,
                image_prompt="chair", config_path=config)
        self.assertEqual(result.asset_paths, {"sam3": self.destination})
        self.assertEqual(result.mesh_paths, {"sam3": mesh})
        self.assertEqual(result.asset_path, self.destination)


if __name__ == "__main__":
    unittest.main()
