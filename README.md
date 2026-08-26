# CAT3DGSPro / HAC 3DGS Compression Evaluation

Perceptual quality evaluation for 3D Gaussian Splatting compression methods
(3DCR, Orig, HACplus+CR, HAC_ori) across the 9 Mip-NeRF360 scenes, at three
compression rates each (high / mid / low). Computes:

- **No-reference image quality**: MUSIQ, MANIQA, CLIP-IQA, NIQE
- **FID** (whole-image, and a patch-based variant following the HiFiC paper)
- **CMMD** (CLIP Maximum Mean Discrepancy, following "Rethinking FID")
- **FID + CMMD with five-crop augmentation**, matching the exact protocol of
  "Drop-In Perceptual Optimization for 3D Gaussian Splatting" (arXiv:2603.23297)

This repo/folder is meant to be portable to another machine. See **Setup**
below for what needs to be installed, and **Data layout** for what the
scripts expect to find on disk.

---

## Setup

Tested with:
- Python 3.12 (conda env name used here: `wd`)
- PyTorch 2.5.1+cu118, CUDA available (developed/run on a single V100 32GB;
  no multi-GPU support, `--device cuda` just uses whichever GPU is default)
- `pyiqa` 0.1.16
- `transformers` 5.15.1

```bash
conda create -n wd python=3.12 -y
conda activate wd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyiqa           # pulls in transformers, accelerate, timm, etc.
```

No other manual installs are needed -- `pyiqa`'s own dependencies cover
everything (`transformers`, `accelerate`, `scipy`, `opencv-python`, etc.).

**First-run downloads (automatic, needs internet):** pretrained weights are
fetched on first use and cached under `~/.cache/torch/hub/pyiqa/` and
`~/.cache/huggingface/`. Expect this on the very first invocation of each
script:
- `musiq_koniq_ckpt-*.pth` (~108MB), `ckpt_koniq10k.pt` (MANIQA, ~543MB),
  `pt_inception-2015-12-05-*.pth` (Inception for FID, ~95MB), plus a
  `timm/vit_base_patch8_224...` backbone (~346MB) used internally by MANIQA.
- `openai/clip-vit-large-patch14-336` (CLIP, for CLIP-IQA and for CMMD).

**Known gotcha -- CLIP checkpoint loading (`eval_cmmd.py`,
`eval_fid_cmmd_fivecrop.py` only):** newer `transformers` refuses to
`torch.load` the original `.bin` weights for `openai/clip-vit-large-patch14-336`
(CVE-2025-32434) unless torch >= 2.6. We're on torch 2.5.1, so both scripts
load a community safetensors conversion via `revision="refs/pr/20"` instead
(see `_CLIP_REVISION` in `eval_cmmd.py`). This is already handled in the code;
mentioned here only so it doesn't look like a mistake if you inspect it, and
in case that PR revision ever disappears from the HF Hub (in which case:
upgrade torch to >=2.6, or find/convert a safetensors copy yourself and point
`_CLIP_MODEL_NAME`/`_CLIP_REVISION` at it).

---

## Data layout the scripts expect

Two independent input trees (paths are CLI defaults in every script; override
with `--cat3dgs_root` / `--hac_root` if they move):

```
/work/u4546465/evaluation/CAT3DGSPro_organized/
    {scene}_{idx:05d}/
        reference.png       <- ground truth for this view
        3DCR@high.png       <- method "3DCR", rate "high"
        3DCR@mid.png
        3DCR@low.png
        Orig@high.png       <- method "Orig", same 3 rates
        Orig@mid.png
        Orig@low.png

/work/u4546465/evaluation/HACplus+CR/HACplus+CR/
    High (0.002)/{scene}/{idx:05d}.png   <- method "HACplus+CR", rate "high"
    Mid (0.007)/{scene}/{idx:05d}.png    <- rate "mid"
    Low (0.05)/{scene}/{idx:05d}.png     <- rate "low"
    HAC_ori/
        High (0.003)/{scene}/{idx:05d}.png  <- method "HAC_ori", rate "high"
        Mid (0.01)/{scene}/{idx:05d}.png    <- rate "mid"
        Low (0.06)/{scene}/{idx:05d}.png    <- rate "low"
```

`scene` is one of the 9 Mip-NeRF360 scenes: `bicycle, bonsai, counter,
flowers, garden, kitchen, room, stump, treehill` (indoor = bonsai/counter/
kitchen/room, outdoor = the other 5 -- this split is used by
`eval_fid_cmmd_fivecrop.py` to match the comparison paper's reporting).

**Important:** the `HACplus+CR` / `HAC_ori` image index order is a
*different* camera-view ordering than `CAT3DGSPro_organized`'s (verified by
direct pixel comparison -- same index does not mean same camera pose across
the two trees). This does **not** matter for anything computed here: NR
metrics don't need a reference at all, and FID/CMMD compare whole *sets* of
images per scene, not paired images, so per-index alignment is irrelevant.
It would matter if you ever add a paired metric (PSNR/SSIM/LPIPS).

All 9 scenes' view counts (fixed, used throughout as a sanity check): bicycle
25, bonsai 37, counter 30, flowers 22, garden 24, kitchen 35, room 39, stump
16, treehill 18 -- 246 total.

To add another method later, edit `EXTERNAL_METHODS` in `eval_cat3dgs.py`
(for anything organized like `HACplus+CR`, i.e. `<root>/<rate_dir>/<scene>/<idx>.png`)
or extend `build_records()` for a different layout.

---

## Scripts

Run everything from `/home/u4546465/iqa_eval/` with the `wd` conda env active:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate wd
cd /home/u4546465/iqa_eval
```

All four eval scripts accept `--limit_scenes bicycle,bonsai --limit_per_scene 3`
for a fast smoke test before committing to a full (~15-60 min) run. Full runs
are slow enough to want backgrounding:
```bash
nohup python eval_cat3dgs.py --device cuda > logs/full_run.log 2>&1 &
```

### 1. `eval_cat3dgs.py` -- no-reference metrics (run this first)

```bash
python eval_cat3dgs.py --device cuda
```
Computes MUSIQ, MANIQA, CLIP-IQA, NIQE (per image) for all 4 methods x 3
rates x 9 scenes. Writes/merges into:
- `results/per_image.csv` -- one row per (method, rate, scene, idx): 4 NR scores
- `results/per_scene.csv` -- one row per (method, rate, scene): mean NR scores
- `results/summary.csv` -- one row per (method, rate), mean NR scores pooled
  across all 9 scenes

Also home to `discover_views()` / `build_records()`, the data-discovery
functions all three other scripts import and reuse -- this file must stay
importable even if you never run its NR metrics yourself.

Useful flags:
- `--methods HAC_ori` -- (re)compute only one method; other methods' existing
  rows in the 3 CSVs above are preserved, not overwritten (see `write_csvs(...,
  merge=True)`). This is how `HAC_ori` was added after the first full run.

**Historical note -- whole-image FID removed:** earlier versions of this
script also computed a whole-image, no-preprocessing FID (via `pyiqa`'s stock
FID on full-resolution images), and `per_scene.csv`/`summary.csv` from before
this change still have a leftover `fid` column with real numbers in it. That
computation was removed from the code (see the module docstring) because at
our sample sizes (16-39 images/scene) it's too unstable to trust -- we
measured this directly: subsampling a 1443-image feature set down to 300
inflated its FID from 16 to 79 with zero change in image content. Use
`eval_fid_patch.py` or `eval_fid_cmmd_fivecrop.py` instead for anything FID
going forward. The existing `fid` column is left in place as historical data
but will not be extended: if this script is ever rerun with `--methods
<something>`, the *rewritten* CSV will no longer have a `fid` column at all
(old rows silently lose that value on rewrite -- `write_csvs()` uses
`extrasaction="ignore"` specifically to allow this without crashing). If you
need that number preserved going forward, keep a copy of the current
`per_scene.csv`/`summary.csv` before rerunning.

**Pooling caveat (applies to every "pooled across scenes" number in this
repo, not just this file):** pooling scenes together for FID/CMMD is **not
the same operation as averaging per-scene values** -- pooling combines very
different scene content into one distribution, and the resulting number is
typically *far below* any individual per-scene value (verified: a pooled FID
can be lower than every one of the 9 per-scene FIDs that went into it,
because compression artifacts push different scenes' feature-space means in
different, partially-cancelling directions). If you need a number to compare
against a specific paper's reported FID/CMMD, check how *that paper*
aggregates across scenes (see `eval_fid_cmmd_fivecrop.py` below for one
paper's example) before assuming a "pooled" number here is the right one to
quote.

### 2. `eval_fid_patch.py` -- patch-based FID (HiFiC-style)

```bash
python eval_fid_patch.py --device cuda
```
Same 4x3x9 grid, but FID is computed on 256x256 patches (two-grid extraction:
non-overlapping grid + grid shifted by 128px, giving 39 patches per 1600x~1050
image) instead of whole images, per Mentzer et al. 2020 ("High-Fidelity
Generative Image Compression") Appendix A.7. This exists because whole-image
FID's sample size problem is severe at our scale (16-39 images/scene); patches
turn that into 624-1521 samples/scene. Runs entirely in memory (no symlink
tree, no patch files written to disk) using a from-scratch reimplementation
of `pyiqa`'s exact FID preprocessing -- validated to match `pyiqa`'s official
FID to 0.0001% relative error before trusting it on real data (see
`validate_against_pyiqa` pattern; there's no automated test for this, it was
a one-off manual check documented in the file's docstring and conversation
history).

Outputs (separate from, does not touch, `eval_cat3dgs.py`'s files):
- `results/fid_patch_per_scene.csv`
- `results/fid_patch_summary.csv` (pooled across all 9 scenes, all 246x39=9594
  patches at once -- same pooling caveat as above applies)

### 3. `eval_cmmd.py` -- CMMD (CLIP Maximum Mean Discrepancy)

```bash
python eval_cmmd.py --device cuda
```
CMMD per Jayasumana et al. 2023 ("Rethinking FID"): CLIP-ViT-L/14-336
embeddings + a Gaussian-RBF Maximum Mean Discrepancy (the **minimum-variance/
biased** MMD estimator specifically -- includes the diagonal `k(x_i,x_i)`
terms, per the official implementation, not the unbiased variant). Operates
on whole images (no patchify -- CMMD doesn't need it the way FID does, since
MMD only needs pairwise kernel similarities, not a full covariance matrix
estimate). Validated against the reference implementation
(`sayakpaul/cmmd-pytorch`, itself a port of the official Google JAX code) by
reproducing their documented example output (`7.696`) exactly before use.

Outputs: `results/cmmd_per_scene.csv`, `results/cmmd_summary.csv` (pooled --
same caveat as FID's pooled number).

### 4. `eval_fid_cmmd_fivecrop.py` -- FID + CMMD matching a specific paper's protocol

```bash
python eval_fid_cmmd_fivecrop.py --device cuda
```
If you need numbers comparable to **"Drop-In Perceptual Optimization for 3D
Gaussian Splatting"** (Ozyilkan, Chen et al., arXiv:2603.23297) specifically,
use this instead of the pooled numbers above. Their Appendix A.4 states:

> "As FID and CMMD require sufficiently large sample sets for stable
> estimation, we augment the evaluation data by extracting five spatial crops
> per view and also adding their horizontal flips." ... "The metrics are then
> computed per scene and averaged across all scenes."

i.e. (a) 5 crops (4 corners + center, `crop_frac=0.75` of the shorter side --
exact crop size isn't specified in the paper, this is a reasonable default,
see `CROP_FRAC` in the script) + horizontal flips = 10 sub-images per view,
fed to **both** FID and CMMD; (b) compute FID/CMMD **per scene**, then take
the **plain arithmetic mean of the 9 per-scene values** -- explicitly *not*
the pool-everything-together approach used elsewhere in this repo. This
distinction is what caused our numbers to initially look "suspiciously
better" than the paper's when compared against the wrong aggregation.

Outputs:
- `results/fivecrop_per_scene.csv` -- one row per (method, rate, scene): fid, cmmd
- `results/fivecrop_summary.csv` -- one row per (method, rate, group), group
  in `{indoor, outdoor, all9}` (indoor/outdoor split matches the paper's
  Mip-NeRF360 categorization) -- **this is the number to quote when comparing
  against that paper's Table 1.**

This script reuses (does not duplicate) the Inception feature extractor from
`eval_fid_patch.py` and the CLIP embedding model from `eval_cmmd.py` via a
`features_for_crops()` / `embed_crops()` method added to each -- both scripts
must be present and importable (same directory) for this one to run.

---

## Result files at a glance

| File | Produced by | Grouping | Notes |
|---|---|---|---|
| `per_image.csv` | eval_cat3dgs.py | every image | NR scores only |
| `per_scene.csv` | eval_cat3dgs.py | method, rate, scene | NR means (plus a frozen/historical `fid` column from an earlier version -- see script's docstring) |
| `summary.csv` | eval_cat3dgs.py | method, rate | NR means, pooled across scenes (plus the same frozen/historical `fid` column) |
| `fid_patch_per_scene.csv` | eval_fid_patch.py | method, rate, scene | patch FID |
| `fid_patch_summary.csv` | eval_fid_patch.py | method, rate | patch FID, **pooled across scenes** |
| `cmmd_per_scene.csv` | eval_cmmd.py | method, rate, scene | CMMD |
| `cmmd_summary.csv` | eval_cmmd.py | method, rate | CMMD, **pooled across scenes** |
| `fivecrop_per_scene.csv` | eval_fid_cmmd_fivecrop.py | method, rate, scene | 5-crop+flip FID & CMMD |
| `fivecrop_summary.csv` | eval_fid_cmmd_fivecrop.py | method, rate, group(indoor/outdoor/all9) | **per-scene averaged**, comparable to arXiv:2603.23297's Table 1 |
| `full_report.csv` | (manual, see conversation) | method, rate, metric | pivoted: one row per metric with a column per scene + a `pooled` column; combines per_scene.csv/summary.csv/fid_patch_*.csv into one sheet-friendly table. Regenerate manually if data changes -- there's no script for this yet, it was built inline with a one-off Python snippet. |

`logs/*.log` are raw stdout from the actual runs that produced the current
`results/` files -- kept for provenance/debugging, not needed to reproduce
anything (rerunning the scripts regenerates everything from scratch).

---

## Metric quick reference (higher/lower is better, what needs a reference)

| Metric | Direction | Needs reference? | Needs paired index? |
|---|---|---|---|
| MUSIQ, MANIQA, CLIP-IQA | higher better | no (no-reference) | n/a |
| NIQE | lower better | no (no-reference) | n/a |
| FID (whole-image, patch, five-crop) | lower better | yes, a whole reference *set* | no (set-vs-set, not pixel-paired) |
| CMMD | lower better | yes, a whole reference *set* | no (set-vs-set, not pixel-paired) |

FID/CMMD are unpaired distribution comparisons -- shuffling render order
within a scene does not change the result, and the `(scene, idx)`
correspondence between a fake image and "its" reference is never used by
these two metrics (it only matters for NR metrics because there is no
reference involved at all, so there's nothing to correspond to).
