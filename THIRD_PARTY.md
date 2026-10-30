# Third-party components

Asset-Forge's MIT licence covers its own orchestration code. It does not replace
the licences/access conditions of downloaded models or software. Weights and
upstream source trees are downloaded into the user's installation, not shipped
in this repository. Output rights depend on the applicable upstream terms.

## Model ownership and licence summary

Reviewed against the primary sources below on **2026-10-30**. This is a
documentation review of the principal model terms, not an exhaustive legal
audit of every transitive dependency. Version-specific licence/notice files
in the downloaded distributions remain authoritative. Model revisions are
recorded by setup; the licence links below may track upstream main branches.

| Component / original developer | Code / model terms | Primary sources |
| --- | --- | --- |
| FLUX.1-dev — Black Forest Labs | FLUX.1-dev Non-Commercial License for the model; output permissions are addressed separately in its terms | [Model card](https://huggingface.co/black-forest-labs/FLUX.1-dev), [model licence](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) |
| FLUX.2-dev — Black Forest Labs | FLUX Non-Commercial License for the model, with separate output/use conditions | [Model card](https://huggingface.co/black-forest-labs/FLUX.2-dev), [model licence](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md) |
| FLUX.2 Klein 4B — Black Forest Labs | Apache 2.0 for this exact optional variant; do not extrapolate to every FLUX variant | [Variant model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| SAM3 — Meta | Custom SAM License, covering SAM materials including code and weights | [Licence](https://github.com/facebookresearch/sam3/blob/main/LICENSE), [gated model](https://huggingface.co/facebook/sam3) |
| Classic SAM — Meta | Apache 2.0 upstream; retain applicable notices | [Licence](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE), [HF checkpoint](https://huggingface.co/facebook/sam-vit-huge) |
| SAM 3D Objects — Meta | Custom SAM License, covering SAM materials including code and weights | [Pinned code licence](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/LICENSE), [gated model](https://huggingface.co/facebook/sam-3d-objects) |
| TRELLIS.2 — Microsoft / TRELLIS.2 authors | MIT code; the TRELLIS.2-4B checkpoint card also declares MIT. DINOv3 and native dependencies have separate terms | [Pinned code licence](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/LICENSE), [checkpoint card](https://huggingface.co/microsoft/TRELLIS.2-4B) |
| DINOv2 — Meta | Apache 2.0 at the pinned source revision; preserve notices | [Pinned licence](https://github.com/facebookresearch/dinov2/blob/b8931f7bf91576930313be2c6d6af376033b35f0/LICENSE) |
| DINOv3 — Meta | Custom DINOv3 License for code/weights, including research-publication acknowledgement requirements | [Licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), [gated encoder](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| MoGe — Microsoft / MoGe authors | MIT code with additional third-party notices in its licence file; the selected moge-vitl checkpoint card declares MIT | [Pinned code licence](https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/LICENSE), [checkpoint card](https://huggingface.co/Ruicheng/moge-vitl) |

Asset-Forge does not train or own these foundation models. It invokes their
published interfaces and downloads their distributions separately. Publications
using the tool should credit Asset-Forge's integration as appropriate **and**
the original models actually used, following each publisher's citation and
acknowledgement instructions. An Asset-Forge citation does not replace them.

Users obtain their own gated access and accept the applicable agreements
themselves. A successful login does not replace model approval. Research use
and an open-source wrapper do not override model terms.

The FLUX.2-dev model card additionally calls for filtering or manual review
under its usage terms. Generated examples in our research workflow are manually
reviewed; the wrapper does not implement a general-purpose content filter or
certify compliance automatically. Consult the current upstream terms before
deploying a different workflow.

The code licence, permission to run a model, and permission to use/distribute
generated images or meshes are separate questions. In particular, a
non-commercial **model** licence does not by itself establish that every output
is non-commercial-only. Read the relevant output clauses and input-data rights
before publishing an asset library or dataset. This repository grants no
blanket licence over generated assets.

## Distribution boundaries

This Git repository contains our integration code, configuration templates and
selected generated example images;
it does not bundle pretrained weights or complete upstream source trees.
Installation fetches those separately and retains their upstream licence files.
MIT cannot be used to relicense downloaded SAM/DINO material or other upstream
components. Preserve existing upstream copyright and notice files, including
when applying the setup's Hydra compatibility patch.

Blender is optional and runs as a separate executable. Blender's distribution
has its own GPL terms; it is not covered by Asset-Forge's MIT licence. The
installer downloads it from the official publisher rather than including it
in the repository. Redistributing a container or environment containing Blender
requires checking the corresponding distribution obligations.

## Installed software

Direct pins and source revisions appear in `environments/` and the two setup
scripts. Resolved package/source inventories are saved in the installation.
Major integrations are listed below; transitive package licences are not
exhaustively classified here and remain with the downloaded distributions.

| Integration | Source/licensing information |
| --- | --- |
| PyTorch / torchvision | [pytorch](https://github.com/pytorch/pytorch) |
| Transformers / Diffusers / Hub / Accelerate | [Hugging Face](https://github.com/huggingface) |
| PyTorch3D | [pytorch3d](https://github.com/facebookresearch/pytorch3d) |
| gsplat | [gsplat](https://github.com/nerfstudio-project/gsplat) |
| Kaolin | [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) |
| nvdiffrast | [nvdiffrast](https://github.com/NVlabs/nvdiffrast) |
| CuMesh / FlexGEMM / utils3d | [CuMesh](https://github.com/JeffreyXiang/CuMesh), [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM), [utils3d](https://github.com/EasternJournalist/utils3d) |
| FlashAttention / xFormers / spconv | [flash-attention](https://github.com/Dao-AILab/flash-attention), [xformers](https://github.com/facebookresearch/xformers), [spconv](https://github.com/traveller59/spconv) |
| Hydra patch | [Pinned upstream file](https://github.com/gleize/hydra/blob/78f00766b5f37672aa7232ebbf01bdd74246bd60/hydra/core/utils.py) |
| Blender | [Licence](https://www.blender.org/about/license/) — invoked as a separate renderer |
| Micromamba | [mamba](https://github.com/mamba-org/mamba) |
| CUDA toolkit | [NVIDIA terms](https://docs.nvidia.com/cuda/eula/index.html) |

Preserve upstream licence and notice files. Redistributing complete environments,
weights or binaries requires a separate review of their applicable obligations.
