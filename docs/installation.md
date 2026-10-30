# Installation and model access

## Support and requirements

Validated model execution: **Linux x86-64, NVIDIA A100 80 GB, on Jean Zay**.
Default FLUX.1/SAM3/SAM3D uses one GPU. The tested FLUX.2-dev BF16 comparison
uses **two A100 80 GB GPUs** and roughly 120 GB of allocated host RAM. These are
tested configurations, not measured minimum requirements.

**Other GPUs have not been tested.** Neither have Windows, macOS or AMD GPUs.
An A100 on another Linux system still needs compatible NVIDIA drivers and a
working native build toolchain. This generalized installer has not yet been
rerun end-to-end on a fresh system; CPU tests do not validate GPU execution.

Bootstrap needs Python >=3.9, Git, Bash, network access, a C++ build-capable
Linux x86-64 machine and substantial storage for multiple CUDA environments and
model checkpoints. Managed environments use Python 3.11. Builds can run without
a GPU present; model execution needs an allocated GPU. Build/download time can
be substantial. No universal disk or host-RAM minimum has been measured.

## Choose the installation folder

```bash
git clone https://github.com/mhurte/Asset-Forge.git
cd Asset-Forge
export AF_INSTALL="/path/on/your/storage/assetforge"
python3 assetgen.py --setup --install-dir "$AF_INSTALL" --environments-only
```

Replace the example path with a writable location.
`--install-dir` does not require a cluster `SCRATCH` variable. When omitted,
setup uses `$SCRATCH` if defined, otherwise `~/.local/share/assetforge`.

Layout:

```text
<install-dir>/venv_for_assetforge/          image + masking runtime
<install-dir>/venv_for_assetforge_sam3d/    SAM3D runtime
<install-dir>/assetforge-tools/            sources, caches, manifests; optional Blender
<install-dir>/venv_for_assetforge_flux2/    optional
<install-dir>/venv_for_assetforge_trellis2/ optional
```

`--env-root "$AF_INSTALL/custom-prefix"` changes the environment prefix inside
the installation directory; tools and other prefixes are placed beside it.
Existing unknown environments are never overwritten. `--resume` continues an
interrupted matching recipe. Use an isolated directory when changing GPU
architectures or FLUX.2 variants.

Native builds default to one compiler worker and CUDA architecture `8.0`
(A100), without GPU auto-detection. Experts trying another NVIDIA architecture
can set `TORCH_CUDA_ARCH_LIST` before setup, but upstream kernels/toolkits may
still be incompatible. Such configurations remain untested.

## Optional Blender previews

Core setup does not download Blender. Add `--with-blender` to your setup command
to install the pinned renderer, including after the base environments already
exist. Alternatively supply `--blender /absolute/path/to/blender` to configure
an existing installation. Repeat the model paths when completing setup.

Normal asset generation exports GLB/PLY without Blender. `--render` and the
default comparison previews require it; `--test --no-render` compares all four
generation routes without rendering. The fresh-install test always requests
Blender, whether or not optional model backends are selected.

## Authentication and model access

Create a read token in [Hugging Face token settings](https://huggingface.co/settings/tokens).
Accept/request access on each applicable model page using the same account:

```bash
"$AF_INSTALL/venv_for_assetforge/bin/hf" auth login
"$AF_INSTALL/venv_for_assetforge/bin/hf" auth whoami
```

Enter the token at the prompt, never in scripts or Git. See the
[official CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

| Component | Official checkpoint/access page |
| --- | --- |
| Default FLUX.1-dev | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Default SAM3 | [SAM3](https://huggingface.co/facebook/sam3) |
| Default SAM3D | [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects) |
| Optional classic SAM | [SAM ViT-H](https://huggingface.co/facebook/sam-vit-huge) |
| Optional FLUX.2-dev | [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| TRELLIS.2 encoder | [DINOv3 ViT-L](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| TRELLIS.2 weights | [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) |

**TRELLIS.2 also needs gated DINOv3 access.** FLUX.2 permission alone is not
sufficient. The default chain is not advertised as login-free. Each publisher's
model access conditions and licences apply independently.

## Download and configure defaults

```bash
HF="$AF_INSTALL/venv_for_assetforge/bin/hf"
"$HF" download black-forest-labs/FLUX.1-dev --local-dir "$AF_INSTALL/models/flux1"
"$HF" download facebook/sam3 --local-dir "$AF_INSTALL/models/sam3"
"$HF" download facebook/sam-3d-objects --local-dir "$AF_INSTALL/models/sam3d"
```

Keep SAM3D's official checkpoint layout and follow its
[upstream setup instructions](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
for auxiliary files referenced by its pipeline. Locate the actual downloaded
`pipeline.yaml`; its subdirectory depends on the upstream checkpoint layout:

```bash
find "$AF_INSTALL/models/sam3d" -name pipeline.yaml
export SAM3D_PIPELINE="/actual/path/from/the/previous/command/pipeline.yaml"
python3 assetgen.py --setup --resume --install-dir "$AF_INSTALL" \
  --flux-model "$AF_INSTALL/models/flux1" \
  --sam3-model "$AF_INSTALL/models/sam3" \
  --sam3d-pipeline "$SAM3D_PIPELINE"
```

Setup resolves checkpoint paths, prepares the offline SAM3D DINOv2 source/weights
cache and writes ignored `assetforge.local.json` in the checkout. Pass
`--config /chosen/deployment.local.json` to both setup and generation to choose
another location. Model directories can be on shared storage outside the install
directory; they must remain available during inference. Inference uses offline
mode and local checkpoints.

For classic SAM, obtain its local HF checkpoint and add
`--sam-model /your/models/sam-vit-huge` to setup. Automatic proposals are unioned
by default; text queries do not select classic SAM masks.

## Optional FLUX.2 and TRELLIS.2

After obtaining permissions, repeat setup with `--with-extras`:

```bash
python3 assetgen.py --setup --resume --with-extras --with-blender --install-dir "$AF_INSTALL" \
  --flux-model "$AF_INSTALL/models/flux1" \
  --sam3-model "$AF_INSTALL/models/sam3" \
  --sam3d-pipeline "$SAM3D_PIPELINE"
```

This installs separate `_flux2`/`_trellis2` environments, compiles extensions and
downloads pinned optional checkpoints. Full TRELLIS/Triton imports are deferred
to an allocated GPU. CPU setup success does not establish GPU kernel support.
`--flux2-variant klein` selects Klein at setup time; only the FLUX.2-dev
two-A100 path has been validated end-to-end. Use an isolated installation to
change variants. DINOv3 access is still required for TRELLIS.2.

## GPU execution

On an allocated GPU, from the checkout:

```bash
python3 assetgen.py --prompt "a blue ceramic teapot" --mask-prompts teapot
```

The full comparison needs two A100 80 GB GPUs and adequate host RAM:

```bash
python3 assetgen.py --test
python3 assetgen.py --test "a robot holding a cup" --mask-prompts robot cup
```

Default outputs go to `$SCRATCH/Asset-forge/TIMESTAMP-JOBID` if `SCRATCH` is set,
otherwise `runs/Asset-forge/TIMESTAMP-local`. Choose a new folder with `--output`.

The coordinator defaults to a 55-minute budget; choose another limit with
`--budget SECONDS`. Manage GPU resources through your own execution environment.
FLUX.2-dev uses balanced two-GPU BF16 placement, recorded beside the image.
Single-GPU CPU offload is available but exceeded a previous 60 GB host-RAM
allowance in testing; it is not the validated route.

## Troubleshooting and validation

- Gated download/401: obtain access and authenticate the same account, then resume.
- Compiler killed: keep `MAX_JOBS=1` or obtain more host RAM for compilation.
- Front-node Triton has no GPU driver: full imports belong in the GPU job.
- Inference OOM: distinguish host RAM from GPU VRAM; inspect worker logs.
- Bad masks, hands or thin geometry: review assets, and supply a corrected mask
  with `--mask` when appropriate. A passed run does not certify visual quality.

The predecessor standalone runner passed an all-four-backend human-action test
on A100s. Public path/setup adaptations have CPU regression coverage; a fresh
full installation and GPU run of this repository are still pending.
