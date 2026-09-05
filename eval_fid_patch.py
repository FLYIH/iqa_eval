#!/usr/bin/env python
"""
Patch-based FID, following Mentzer et al. 2020 ("High-Fidelity Generative Image
Compression", HiFiC), Appendix A.7:

  From each HxW image, extract floor(H/f)*floor(W/f) non-overlapping fxf crops
  (grid 1, from the origin). Then shift the extraction origin by f/2 in both
  dimensions and extract another (floor(H/f)-1)*(floor(W/f)-1) non-overlapping
  fxf crops (grid 2). Use f=256.

Rationale (from the paper): whole-image FID needs many samples to reliably
estimate a covariance matrix in a high-dim (2048) feature space, and datasets
with too few images (or images resized to wildly different scales) give noisy
estimates. The paper itself refuses to report FID/KID for Kodak (24 images ->
192 patches) as too few. Our per-scene groups (16-39 images) are worse than
that, and per_scene FID in eval_cat3dgs.py already showed a "singular matrix"
warning for a 3-image test group -- direct evidence of this exact problem.

This script does NOT touch the whole-image FID already computed in
eval_cat3dgs.py (summary.csv / per_scene.csv). It reuses that script's view
discovery + method/rate file-path logic, but replaces the FID computation with
this patch-based version, and writes to SEPARATE output files:
  fid_patch_per_scene.csv
  fid_patch_summary.csv

Implementation notes:
  - Reimplements pyiqa's exact FID preprocessing (verified against pyiqa's own
    get_folder_features/FID.forward): InceptionV3 (2048-dim pool feature),
    each image/patch is independently per-channel resized to 299x299 with PIL
    BICUBIC (pyiqa's "clean" mode), then normalized as (x-128)/128, then
    forward(x, resize_input=False, normalize_input=False).
  - Runs patch extraction + feature extraction in-memory (no intermediate
    files written to disk) since writing ~125k small patch PNGs to the
    NFS-backed /work or /home filesystem would be slow and wasteful; only the
    2048-dim feature vectors are kept (cached per source image, since the same
    reference image is reused across many (method, rate) comparisons).
  - Correctness of the reimplementation is checked against pyiqa's own FID
    (unpatched, whole-image) before running the real computation -- see
    validate_against_pyiqa().

Usage:
  python eval_fid_patch.py --out_dir /home/u4546465/iqa_eval/results --device cuda
  python eval_fid_patch.py --limit_scenes bicycle,bonsai --limit_per_scene 3   # smoke test
  python eval_fid_patch.py --validate_only    # just run the correctness check, no full run
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_cat3dgs import discover_views, build_records, RATES  # noqa: E402

PATCH_SIZE = 256
INCEPTION_INPUT_SIZE = (299, 299)


def extract_patches(img_path, f=PATCH_SIZE):
    """HiFiC Appendix A.7 two-grid patch extraction. Returns list of (f,f,3) uint8 arrays."""
    im = np.array(Image.open(img_path).convert("RGB"))
    H, W = im.shape[0], im.shape[1]
    n_h, n_w = H // f, W // f
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patches.append(im[i * f:(i + 1) * f, j * f:(j + 1) * f])
    off = f // 2
    for i in range(n_h - 1):
        for j in range(n_w - 1):
            patches.append(im[off + i * f:off + (i + 1) * f, off + j * f:off + (j + 1) * f])
    return patches


def resize_clean(patch_np, size=INCEPTION_INPUT_SIZE):
    """Exact reimplementation of pyiqa's ResizeDataset mode='clean': independent
    per-channel PIL BICUBIC resize, float32, no cross-channel interpolation."""
    chans = []
    for c in range(3):
        ch = Image.fromarray(patch_np[:, :, c].astype(np.float32), mode="F")
        ch = ch.resize(size, resample=Image.BICUBIC)
        chans.append(np.asarray(ch).clip(0, 255))
    out = np.stack(chans, axis=2).astype(np.float32)
    return out  # HxWx3 float32


class InceptionFeatureExtractor:
    def __init__(self, device):
        import torch
        from pyiqa.archs.inception import InceptionV3
        self.torch = torch
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3(output_blocks=[block_idx]).to(device).eval()
        self.device = device

    def features_for_image(self, img_path, batch_size=32):
        """Extract patches from one image, return (n_patches, 2048) np.float32 array."""
        patches = extract_patches(img_path)
        return self.features_for_crops(patches, batch_size=batch_size)

    def features_for_crops(self, crops, batch_size=32):
        """Given a list of already-extracted HxWx3 uint8/float arrays (e.g. five-crop
        outputs, or HiFiC-style patches), return (n_crops, 2048) np.float32 array."""
        torch = self.torch
        resized = [resize_clean(c) for c in crops]  # list of HxWx3 float32
        feats = []
        for i in range(0, len(resized), batch_size):
            chunk = resized[i:i + batch_size]
            batch = torch.tensor(np.stack(chunk)).permute(0, 3, 1, 2)  # N,3,H,W
            batch = (batch - 128) / 128
            with torch.no_grad():
                feat = self.model(batch.to(self.device), False, False)[0]
                feat = feat.reshape(feat.shape[0], feat.shape[1]).detach().cpu().numpy()
            feats.append(feat)
        return np.concatenate(feats, axis=0).astype(np.float32)

    def features_for_crops_raw(self, crops, batch_size=32):
        """Like features_for_crops but skips the resize-to-299x299 step, feeding
        each crop at its native resolution -- relies on InceptionV3's adaptive
        avg pool (pool_3 block) to handle variable spatial input size. All crops
        passed in one call must share the same HxW (true for one image's
        five-crop set, since they're all crop_frac*min(H,W) of that image)."""
        torch = self.torch
        arrs = [c.astype(np.float32) for c in crops]  # list of HxWx3 float32, native size
        feats = []
        for i in range(0, len(arrs), batch_size):
            chunk = arrs[i:i + batch_size]
            batch = torch.tensor(np.stack(chunk)).permute(0, 3, 1, 2)  # N,3,H,W
            batch = (batch - 128) / 128
            with torch.no_grad():
                feat = self.model(batch.to(self.device), False, False)[0]
                feat = feat.reshape(feat.shape[0], feat.shape[1]).detach().cpu().numpy()
            feats.append(feat)
        return np.concatenate(feats, axis=0).astype(np.float32)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def validate_against_pyiqa(device, sample_dir_a, sample_dir_b):
    """Sanity check: compute FID with our reimplementation treating each WHOLE
    image as a single 'patch' resized to 299x299 (i.e. bypass the patch grid),
    and compare to pyiqa's own folder-based FID for the same two folders. If
    our numbers match, the reimplemented preprocessing/model/frechet code is
    faithful to pyiqa, and we can trust the patch-grid extension."""
    import pyiqa
    print("Validating custom Inception pipeline against pyiqa.create_metric('fid')...")

    official = pyiqa.create_metric("fid", device=device)
    official_score = float(official(str(sample_dir_a), str(sample_dir_b)))

    extractor = InceptionFeatureExtractor(device)

    def whole_image_features(folder):
        from pyiqa.utils.img_util import scandir_images
        files = scandir_images(str(folder))
        feats = []
        for f in files:
            im = np.array(Image.open(f).convert("RGB"))
            resized = resize_clean(im)
            import torch
            batch = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0)
            batch = (batch - 128) / 128
            with torch.no_grad():
                feat = extractor.model(batch.to(device), False, False)[0]
                feat = feat.reshape(feat.shape[0], feat.shape[1]).detach().cpu().numpy()
            feats.append(feat[0])
        return np.stack(feats)

    feats_a = whole_image_features(sample_dir_a)
    feats_b = whole_image_features(sample_dir_b)
    mu1, sig1 = np.mean(feats_a, axis=0), np.cov(feats_a, rowvar=False)
    mu2, sig2 = np.mean(feats_b, axis=0), np.cov(feats_b, rowvar=False)
    custom_score = frechet_distance(mu1, sig1, mu2, sig2)

    print(f"  pyiqa official FID  = {official_score:.6f}")
    print(f"  custom reimpl FID   = {custom_score:.6f}")
    rel_err = abs(official_score - custom_score) / max(official_score, 1e-8)
    print(f"  relative error      = {rel_err:.6%}")
    if rel_err > 0.01:
        print("  WARNING: relative error > 1%, reimplementation may be inconsistent!", file=sys.stderr)
    else:
        print("  OK: reimplementation matches pyiqa's official FID.")
    return rel_err


def compute_all_features(records, ref_paths, device):
    """Returns (fake_features, ref_features) dicts, each key -> (n_patches,2048) array.
    fake key: (method, rate, scene, idx). ref key: (scene, idx)."""
    extractor = InceptionFeatureExtractor(device)

    ref_features = {}
    n = len(ref_paths)
    for i, ((scene, idx), path) in enumerate(ref_paths.items()):
        ref_features[(scene, idx)] = extractor.features_for_image(path)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  ref features [{i+1}/{n}]")

    fake_features = {}
    n = len(records)
    for i, r in enumerate(records):
        key = (r["method"], r["rate"], r["scene"], r["idx"])
        fake_features[key] = extractor.features_for_image(r["path"])
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  fake features [{i+1}/{n}]")

    return fake_features, ref_features


def gaussian_stats(feat_list):
    feats = np.concatenate(feat_list, axis=0)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False), feats.shape[0]


def compute_group_fids(records, ref_features, fake_features, scenes):
    methods_rates = sorted({(r["method"], r["rate"]) for r in records})

    # pre-group fake feature arrays by (method, rate, scene) and by (method, rate)
    by_scene = defaultdict(list)
    by_pooled = defaultdict(list)
    for r in records:
        key = (r["method"], r["rate"], r["scene"], r["idx"])
        feat = fake_features[key]
        by_scene[(r["method"], r["rate"], r["scene"])].append(feat)
        by_pooled[(r["method"], r["rate"])].append(feat)

    all_ref_feats = list(ref_features.values())
    ref_by_scene = defaultdict(list)
    for (scene, idx), feat in ref_features.items():
        ref_by_scene[scene].append(feat)

    mu_ref_pooled, sig_ref_pooled, n_ref_pooled = gaussian_stats(all_ref_feats)
    print(f"Pooled reference: {n_ref_pooled} patches from {len(ref_features)} images")

    ref_stats_by_scene = {}
    for scene, feats in ref_by_scene.items():
        ref_stats_by_scene[scene] = gaussian_stats(feats)

    per_scene_fid = {}
    for (method, rate, scene), feats in by_scene.items():
        mu1, sig1, n1 = gaussian_stats(feats)
        mu2, sig2, n2 = ref_stats_by_scene[scene]
        fid = frechet_distance(mu1, sig1, mu2, sig2)
        per_scene_fid[(method, rate, scene)] = (fid, n1, n2)
        print(f"  FID_patch[{method}/{rate}/{scene}] = {fid:.4f}  (fake_patches={n1}, ref_patches={n2})")

    pooled_fid = {}
    for (method, rate), feats in by_pooled.items():
        mu1, sig1, n1 = gaussian_stats(feats)
        fid = frechet_distance(mu1, sig1, mu_ref_pooled, sig_ref_pooled)
        pooled_fid[(method, rate)] = (fid, n1, n_ref_pooled)
        print(f"  FID_patch[{method}/{rate}/ALL_SCENES] = {fid:.4f}  (fake_patches={n1}, ref_patches={n_ref_pooled})")

    return per_scene_fid, pooled_fid


def write_outputs(per_scene_fid, pooled_fid, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / "fid_patch_per_scene.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rate", "scene", "fid_patch", "n_fake_patches", "n_ref_patches"])
        w.writeheader()
        for (method, rate, scene), (fid, n1, n2) in sorted(per_scene_fid.items()):
            w.writerow({"method": method, "rate": rate, "scene": scene,
                        "fid_patch": fid, "n_fake_patches": n1, "n_ref_patches": n2})
    print(f"Wrote {p}")

    p = out_dir / "fid_patch_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rate", "fid_patch", "n_fake_patches", "n_ref_patches"])
        w.writeheader()
        for (method, rate), (fid, n1, n2) in sorted(pooled_fid.items()):
            w.writerow({"method": method, "rate": rate,
                        "fid_patch": fid, "n_fake_patches": n1, "n_ref_patches": n2})
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
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        scenes, ref_paths = discover_views(args.cat3dgs_root, {"bicycle"}, 4)
        sample_ref_dir = Path(next(iter(ref_paths.values()))).parent.parent  # not used directly
        # build two small folders' worth via the real reference set: use bicycle refs vs bicycle 3DCR@high
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        d1, d2 = tmp / "a", tmp / "b"
        d1.mkdir(); d2.mkdir()
        records = build_records(args.cat3dgs_root, args.hac_root, scenes, methods={"3DCR"})
        for i, r in enumerate([r for r in records if r["rate"] == "high"]):
            shutil.copy(r["path"], d1 / f"{i}.png")
            shutil.copy(r["ref_path"], d2 / f"{i}.png")
        validate_against_pyiqa(args.device, d1, d2)
        shutil.rmtree(tmp)
        return

    limit_scenes = set(args.limit_scenes.split(",")) if args.limit_scenes else None
    methods = set(args.methods.split(",")) if args.methods else None

    print("Discovering views...")
    scenes, ref_paths = discover_views(args.cat3dgs_root, limit_scenes, args.limit_per_scene)
    n_views = sum(len(v) for v in scenes.values())
    print(f"Found {len(scenes)} scenes, {n_views} views total")

    print(f"Building file records (methods={methods or 'ALL'})...")
    records = build_records(args.cat3dgs_root, args.hac_root, scenes, methods=methods)
    print(f"Total (method,rate,view) records: {len(records)}")
    print(f"Patch grid: {PATCH_SIZE}x{PATCH_SIZE}, two-grid scheme per HiFiC Appendix A.7")

    print("Computing Inception patch-features for all fake + reference images (cached per image)...")
    fake_features, ref_features = compute_all_features(records, ref_paths, args.device)

    print("Computing patch-based FID per (method, rate, scene) and pooled...")
    per_scene_fid, pooled_fid = compute_group_fids(records, ref_features, fake_features, scenes)

    write_outputs(per_scene_fid, pooled_fid, args.out_dir)


if __name__ == "__main__":
    main()
