# Validation record

Validation review, 2026-10-30:

- 22 standalone/configuration/installation tests passed.
- 18 model-adapter/export/mask-retry tests passed with mocked model execution.
- Python 3.9 syntax checks passed for all 28 tracked Python scripts/tests.
- Personal-path and credential-pattern scans found no matches in the tracked
  source snapshot and its development history at the time of review.
  This is a bounded pattern scan, not a guarantee against all possible secrets.
- All local Markdown/image links resolved. Tracked text decoded as UTF-8 and
  JSON parsed successfully. All 16 example PNGs verified; exported images have
  no embedded File, Hostname or EXIF fields. Example pixels were verified
  unchanged during export.
- The auxiliary TRELLIS-image-large checkpoint is now included in the early
  access check, alongside the principal models and DINOv3 encoder.

## Hardware scope

Predecessor model execution was tested on Jean Zay A100s. The normal installer
accepts `--install-dir` independently of cluster paths. Other GPU architectures
remain untested; CUDA builds default to A100 architecture 8.0 unless an expert
explicitly chooses another build target.

## Outstanding validation

The predecessor standalone code passed a four-backend human-action comparison
on A100s. The generic installer, authentication preflight, optional
Blender setup and mask-retry changes have CPU regression coverage. Those checks
do not install CUDA environments or execute model weights.

The fresh-release installation attempt reached the access probe and correctly
stopped on missing gated-model permissions. A successful complete fresh
installation and subsequent offline A100 comparison have not yet been confirmed.
Keep that distinction visible when describing support.

Record the tested commit and actual GPU results here after completing those
checks.

The root MIT licence covers our code. Model/dependency terms and generated-media
scope are documented separately; this review is not a complete legal audit of
all transitive dependencies or a hardware compatibility certification.
