#!/usr/bin/env python
"""
CMMD (CLIP Maximum Mean Discrepancy), following Jayasumana et al. 2023,
"Rethinking FID: Towards a Better Evaluation Metric for Image Generation"
(https://github.com/google-research/google-research/tree/master/cmmd).

Motivation: FID assumes Inception features are Gaussian-distributed and its
estimator is biased for small sample sizes (we measured this directly: the
same 1443-patch set, subsampled down to 300, saw its FID jump from 16 to 79
with zero change in image content -- see conversation history). CMMD instead
uses CLIP embeddings (which capture higher-level semantics than Inception)
and the minimum-variance (biased) Maximum Mean Discrepancy (MMD) estimator
used by the official implementation, with a Gaussian RBF kernel -- no Gaussianity assumption, and much less
sensitive to sample size.

This reimplements the unofficial PyTorch port (sayakpaul/cmmd-pytorch, which
itself is a faithful port of the official JAX code) using `transformers`
instead of `scenic`, since we already have `transformers` installed as a
pyiqa dependency and did not want to add JAX+Scenic. The CLIP checkpoint
(openai/clip-vit-large-patch14-336, i.e. ViT-L/14@336px, exactly what the
paper uses) is loaded via a community safetensors-converted revision
(revision="refs/pr/20") because transformers now refuses to torch.load the
original .bin checkpoint on our torch version (CVE-2025-32434) -- verified
this loads byte-identical weights (only the unused text-tower keys differ,
which is expected since we only load CLIPVisionModelWithProjection).

Correctness of this whole pipeline (CLIP embedding + MMD formula) was
validated against the reference repo's own documented example before being
applied to real data: running the repo's unmodified example images through
this exact code path reproduces their documented value of 7.696 exactly.

CMMD, like FID, operates on whole images (not patches) per the paper -- CLIP's
crop-to-336x336 preprocessing already discards a lot of the FID patch-vs-whole
image resolution concern, and the paper's own experiments use whole images.

Like FID, CMMD is an unpaired, distribution-level metric: it never uses the
(scene, idx) correspondence between a fake image and its matching reference --
it only compares the fake image *set* against the reference image *set* for
that group. Shuffling the render order within a scene does not change the
result. This is a deliberate difference from paired metrics (PSNR/SSIM/LPIPS,
not computed here): those answer "does view i look like its own reference?",
CMMD/FID answer "does this whole set of views look like it came from the same
distribution as the reference set?".

Pooled (all-scenes) CMMD is the closest analogue to the paper's own
dataset-level protocol and should be treated as the primary number; per-scene
CMMD is a useful breakdown but each scene has far fewer images (16-39) than
the pooled set (246), so per-scene values carry more sampling noise -- though
this is a much smaller concern for CMMD than for FID, since CMMD only needs
pairwise kernel similarities, not a full 2048-dim covariance matrix estimate.

Usage:
  python eval_cmmd.py --out_dir /home/u4546465/iqa_eval/results --device cuda
  python eval_cmmd.py --limit_scenes bicycle,bonsai --limit_per_scene 3   # smoke test
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_cat3dgs import discover_views, build_records  # noqa: E402

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"
_CLIP_REVISION = "refs/pr/20"  # safetensors conversion; main branch is torch.load-only (blocked)
_SIGMA = 10
_SCALE = 1000


class ClipEmbeddingModel:
    """Port of cmmd-pytorch's embedding.ClipEmbeddingModel."""

    def __init__(self, device):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.image_processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_NAME)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            _CLIP_MODEL_NAME, revision=_CLIP_REVISION
        ).eval().to(device)
        self.device = device
        self.input_image_size = self.image_processor.crop_size["height"]

    def _center_crop_and_resize(self, im, size):
        w, h = im.size
        l = min(w, h)
        top, left = (h - l) // 2, (w - l) // 2
        im = im.crop((left, top, left + l, top + l))
        return im.resize((size, size), resample=Image.BICUBIC)

    @torch.no_grad()
    def embed_image(self, img_path):
        im = Image.open(img_path).convert("RGB")
        im = self._center_crop_and_resize(im, self.input_image_size)
        return self.embed_crops([np.asarray(im)])[0]

    @torch.no_grad()
    def embed_crops(self, crops, batch_size=32):
        """Given a list of already-extracted HxWx3 uint8 arrays (e.g. five-crop
        outputs), resize each to the CLIP input size and return (n_crops, embed_dim)."""
        embs = []
        for i in range(0, len(crops), batch_size):
            chunk = crops[i:i + batch_size]
            resized = [np.asarray(Image.fromarray(c).resize(
                (self.input_image_size, self.input_image_size), resample=Image.BICUBIC
            )) for c in chunk]
            arr = np.stack(resized).astype(np.float32) / 255.0  # NxHxWx3, [0,1]
            inputs = self.image_processor(
                images=arr, do_normalize=True, do_center_crop=False,
                do_resize=False, do_rescale=False, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = self._model(**inputs).image_embeds.cpu()
            emb = emb / torch.linalg.norm(emb, dim=-1, keepdim=True)
            embs.append(emb.numpy())
        return np.concatenate(embs, axis=0)  # (n_crops, embedding_dim)

    @torch.no_grad()
    def embed_crops_raw(self, crops, batch_size=32):
        """Like embed_crops but skips resizing to input_image_size (336x336):
        feeds each crop at native resolution with interpolate_pos_encoding=True
        so CLIP's fixed position embeddings are bicubic-interpolated to match,
        instead of requiring an exact 336x336 input. All crops passed in one
        call must share the same HxW (true for one image's five-crop set)."""
        embs = []
        for i in range(0, len(crops), batch_size):
            chunk = crops[i:i + batch_size]
            arr = np.stack(chunk).astype(np.float32) / 255.0  # NxHxWx3, [0,1], native size
            inputs = self.image_processor(
                images=arr, do_normalize=True, do_center_crop=False,
                do_resize=False, do_rescale=False, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = self._model(**inputs, interpolate_pos_encoding=True).image_embeds.cpu()
            emb = emb / torch.linalg.norm(emb, dim=-1, keepdim=True)
            embs.append(emb.numpy())
        return np.concatenate(embs, axis=0)  # (n_crops, embedding_dim)


def mmd(x, y):
    """Exact port of cmmd-pytorch's distance.mmd (sigma=10, scale=1000)."""
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))
    gamma = 1 / (2 * _SIGMA ** 2)
    k_xx = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + x_sqnorms[:, None] + x_sqnorms[None, :])))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + x_sqnorms[:, None] + y_sqnorms[None, :])))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + y_sqnorms[:, None] + y_sqnorms[None, :])))
    return float(_SCALE * (k_xx + k_yy - 2 * k_xy))


def compute_all_embeddings(records, ref_paths, device):
    model = ClipEmbeddingModel(device)

    ref_embs = {}
    n = len(ref_paths)
    for i, (key, path) in enumerate(ref_paths.items()):
        ref_embs[key] = model.embed_image(path)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  ref embeddings [{i+1}/{n}]")

    fake_embs = {}
    n = len(records)
    for i, r in enumerate(records):
        key = (r["method"], r["rate"], r["scene"], r["idx"])
        fake_embs[key] = model.embed_image(r["path"])
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  fake embeddings [{i+1}/{n}]")

    return fake_embs, ref_embs


def compute_group_cmmd(records, ref_embs, fake_embs, scenes):
    by_scene = defaultdict(list)
    by_pooled = defaultdict(list)
    for r in records:
        key = (r["method"], r["rate"], r["scene"], r["idx"])
        emb = fake_embs[key]
        by_scene[(r["method"], r["rate"], r["scene"])].append(emb)
        by_pooled[(r["method"], r["rate"])].append(emb)

    ref_by_scene = defaultdict(list)
    for (scene, idx), emb in ref_embs.items():
        ref_by_scene[scene].append(emb)
    ref_pooled = np.stack(list(ref_embs.values()))
    print(f"Pooled reference: {ref_pooled.shape[0]} images")

    per_scene_cmmd = {}
    for (method, rate, scene), embs in by_scene.items():
        fake_arr = np.stack(embs)
        ref_arr = np.stack(ref_by_scene[scene])
        val = mmd(fake_arr, ref_arr)
        per_scene_cmmd[(method, rate, scene)] = (val, len(embs), len(ref_by_scene[scene]))
        print(f"  CMMD[{method}/{rate}/{scene}] = {val:.4f}  (n_fake={len(embs)}, n_ref={len(ref_by_scene[scene])})")

    pooled_cmmd = {}
    for (method, rate), embs in by_pooled.items():
        fake_arr = np.stack(embs)
        val = mmd(fake_arr, ref_pooled)
        pooled_cmmd[(method, rate)] = (val, len(embs), ref_pooled.shape[0])
        print(f"  CMMD[{method}/{rate}/ALL_SCENES] = {val:.4f}  (n_fake={len(embs)}, n_ref={ref_pooled.shape[0]})")

    return per_scene_cmmd, pooled_cmmd


def write_outputs(per_scene_cmmd, pooled_cmmd, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / "cmmd_per_scene.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rate", "scene", "cmmd", "n_fake", "n_ref"])
        w.writeheader()
        for (method, rate, scene), (val, n1, n2) in sorted(per_scene_cmmd.items()):
            w.writerow({"method": method, "rate": rate, "scene": scene, "cmmd": val, "n_fake": n1, "n_ref": n2})
    print(f"Wrote {p}")

    p = out_dir / "cmmd_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rate", "cmmd", "n_fake", "n_ref"])
        w.writeheader()
        for (method, rate), (val, n1, n2) in sorted(pooled_cmmd.items()):
            w.writerow({"method": method, "rate": rate, "cmmd": val, "n_fake": n1, "n_ref": n2})
    print(f"Wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat3dgs_root", default="/work/u4546465/evaluation/CAT3DGSPro_organized")
    ap.add_argument("--hac_root", default="/work/u4546465/evaluation/HACplus+CR")
    ap.add_argument("--out_dir", default="/home/u4546465/iqa_eval/results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_scenes", default=None)
    ap.add_argument("--limit_per_scene", type=int, default=None)
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()

    limit_scenes = set(args.limit_scenes.split(",")) if args.limit_scenes else None
    methods = set(args.methods.split(",")) if args.methods else None

    print("Discovering views...")
    scenes, ref_paths = discover_views(args.cat3dgs_root, limit_scenes, args.limit_per_scene)
    print(f"Found {len(scenes)} scenes, {sum(len(v) for v in scenes.values())} views total")

    print(f"Building file records (methods={methods or 'ALL'})...")
    records = build_records(args.cat3dgs_root, args.hac_root, scenes, methods=methods)
    print(f"Total (method,rate,view) records: {len(records)}")

    print("Computing CLIP embeddings for all fake + reference images (cached per image)...")
    fake_embs, ref_embs = compute_all_embeddings(records, ref_paths, args.device)

    print("Computing CMMD per (method, rate, scene) and pooled...")
    per_scene_cmmd, pooled_cmmd = compute_group_cmmd(records, ref_embs, fake_embs, scenes)

    write_outputs(per_scene_cmmd, pooled_cmmd, args.out_dir)


if __name__ == "__main__":
    main()
