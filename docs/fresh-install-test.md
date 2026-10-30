# Fresh-install test

Run online on a Linux x86-64 build/login node before reserving GPUs. This test
includes the optional Blender renderer automatically and uses a new installation directory and freshly downloads default checkpoints;
it does not adopt personal paths or the previous working environments.

```bash
git clone https://github.com/mhurte/Asset-Forge.git Asset-Forge-release-test
cd Asset-Forge-release-test
export AF_TEST="/path/to/new/assetforge-install-test"
python3 install_test.py --install-dir "$AF_TEST" --with-extras
```

The test bootstraps a small Hugging Face client environment, reuses your existing
saved login or `HF_TOKEN`, and checks read access for each required model before
downloading large weights or compiling CUDA extensions. It includes FLUX.1,
SAM3, SAM3D, classic SAM, MoGe, FLUX.2-dev, TRELLIS.2, its auxiliary
`microsoft/TRELLIS-image-large` checkpoint and its DINOv3 encoder.
Omit `--with-extras` to test only the default stack and classic SAM.

Before downloading weights, accept/request access on each applicable model
page using the **same Hugging Face account as your read token**:

| Setup | Model approvals |
| --- | --- |
| Default pipeline | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), [SAM3](https://huggingface.co/facebook/sam3), [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects) |
| Optional FLUX.2-dev | [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) |
| Optional TRELLIS.2 | [DINOv3 encoder](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), in addition to the [TRELLIS.2 checkpoint terms](https://huggingface.co/microsoft/TRELLIS.2-4B) |

Creating a token or logging in does not grant model access. Downloads cannot
proceed until the account has the required approval and the token permits read
access to those repositories. Fine-grained tokens must include the relevant
gated-repository permissions. The installer cannot accept agreements or obtain
approval for you. Enter tokens only at the interactive login prompt.

If no valid login is available, use the same command with `--resume --login`.
The login is interactive and is not redirected to the installation log. Never
include a token as a command-line argument. The same Hugging Face cache settings
are inherited by the subsequent environments. No token is copied into reports,
deployment configs or this repository.

If a model is gated and your account is not approved, the probe reports its
model page and stops before the expensive build. Open that page, accept/request
access yourself, wait for approval, then repeat with `--resume`. A valid token
does not substitute for model approval or fine-grained token read permissions.

After approval, resume the same test directory without rebuilding the
bootstrap environment:

```bash
python3 install_test.py --install-dir "$AF_TEST" --with-extras --resume
```

To check permissions first without downloading model weights or building CUDA:

```bash
python3 install_test.py --install-dir "$AF_TEST" --with-extras --access-only
```

Then rerun without `--access-only`, adding `--resume`. Keep the same extras
selection when resuming. Bootstrap package installation still needs internet
and a Python installation with working `venv`/`ensurepip` support.

The test saves `install-test.json`, `model-access.json` (statuses/revisions, no
token) and per-stage logs under `$AF_TEST`. Runtime compilation writes to
`$AF_TEST/runtime-setup.log`; inspect that file in another terminal while it runs.
Interrupted matching builds/downloads can resume. This is an attempted clean
installation, not a guarantee that network, upstream packages or kernels cannot
fail.

## GPU smoke test after installation succeeds

On a Linux machine with access to two A100 80 GB GPUs and sufficient host RAM,
run from the checkout after the online installation has completed:

```bash
python3 assetgen.py --test \
  --config "$AF_TEST/deployment.local.json" \
  --output "$AF_TEST/gpu-cup-test"
```

For a default-stack installation without optional model backends, use one A100:

```bash
python3 assetgen.py --config "$AF_TEST/deployment.local.json" \
  --prompt "a yellow ceramic mug on a white background" --mask-prompts mug \
  --render --output "$AF_TEST/gpu-cup-test"
```

The full installation test includes Blender for these previews. No login or
agreement acceptance happens during inference; all model downloads must finish
on the online machine first. Review both the installation report and GPU gallery.
Installation success alone leaves GPU validation pending. Use a new output folder
for another GPU run. GPU access and resource scheduling are managed by the user.
