# Low-level adapters

The supported entry point is `python3 assetgen.py` in the repository root.
See the [installation guide](../../docs/installation.md).

These modules also expose `generate_image`, `generate_mask` and
`generate_3d_asset` for Python callers. JSON files in this directory are templates
with `/path/to` placeholders. Run `--setup` to generate usable local configs,
or supply your own `config_path` when calling an adapter.

The CLI chooses one masking backend. SAM3 unions the masks for each requested
participant; classic SAM unions accepted automatic proposals. The legacy
`asset_generation_pipeline.py` may attempt both backends if both are enabled.
SAM3D exports Gaussian PLY and GLB; TRELLIS.2 exports a textured mesh GLB.
