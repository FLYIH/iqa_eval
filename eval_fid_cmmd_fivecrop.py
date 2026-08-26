#!/usr/bin/env python
"""
FID + CMMD computed the way "Drop-In Perceptual Optimization for 3D Gaussian
Splatting" (Ozyilkan, Chen et al., arXiv:2603.23297) does it, per their
Appendix A.4:

  "As FID and CMMD require sufficiently large sample sets for stable
  estimation, we augment the evaluation data by extracting five spatial
  crops per view and also adding their horizontal flips."

  "The metrics are then computed per scene and averaged across all scenes
  to ensure statistical stability while accounting for inter-scene
  variation." -- i.e. compute FID/CMMD separately for each scene (fake
  crops of that scene vs. reference crops of that scene), THEN take the
  plain arithmetic mean of the per-scene values. This is NOT the same as
  pooling all scenes into one distribution (which is what our earlier
  summary.csv / cmmd_summary.csv "pooled" numbers do) -- pooling across
  visually very different scenes causes destructive cancellation and
  produces numbers far below any individual scene's value (verified
  empirically earlier: pooled patch-FID for 3DCR/high was 6.35, while every
  one of its 9 per-scene values was >= 14.38). The paper's own reported
  Table 1 numbers are the per-scene-averaged kind, which is why comparing
  our pooled numbers against their table looked "suspiciously good" until
  this was corrected.

Five-crop: the paper doesn't specify exact crop size/positions, so this
implements the standard "FiveCrop" convention (torchvision.transforms.FiveCrop):
4 corner crops + 1 center crop, each of size `crop_frac * min(H, W)` (default
0.75), plus a horizontal flip of each = 10 sub-images per view fed to both
FID (Inception-v3, via eval_fid_patch.InceptionFeatureExtractor) and CMMD
(CLIP-ViT-L/14-336, via eval_cmmd.ClipEmbeddingModel) -- the same augmented
crops are shared between both metrics, matching the paper's description of a
single augmented evaluation set feeding both.

Outputs (does not touch any existing CSV):
  fivecrop_per_scene.csv    one row per (method, rate, scene): fid, cmmd
  fivecrop_summary.csv      one row per (method, rate, group) where group in
                            {indoor, outdoor, all9}: mean of per-scene fid/cmmd
                            (indoor = bonsai/counter/kitchen/room,
                             outdoor = bicycle/flowers/garden/stump/treehill,
                             matching the paper's Mip-NeRF360 indoor/outdoor split)

Usage:
  python eval_fid_cmmd_fivecrop.py --out_dir /home/u4546465/iqa_eval/results --device cuda
  python eval_fid_cmmd_fivecrop.py --limit_scenes bicycle,bonsai --limit_per_scene 3   # smoke test
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_cat3dgs import discover_views, build_records  # noqa: E402
from eval_fid_patch import InceptionFeatureExtractor, frechet_distance  # noqa: E402
from eval_cmmd import ClipEmbeddingModel, mmd  # noqa: E402

INDOOR = {"bonsai", "counter", "kitchen", "room"}
OUTDOOR = {"bicycle", "flowers", "garden", "stump", "treehill"}
CROP_FRAC = 0.75


def five_crop_and_flip(img_path, crop_frac=CROP_FRAC):
    """Standard FiveCrop (4 corners + center) at size crop_frac*min(H,W), plus a
    horizontal flip of each -> 10 HxWx3 uint8 arrays per image."""
    im = np.array(Image.open(img_path).convert("RGB"))
    H, W = im.shape[0], im.shape[1]
    c = int(round(crop_frac * min(H, W)))
    boxes = [
        (0, 0), (0, W - c), (H - c, 0), (H - c, W - c),  # TL, TR, BL, BR
        ((H - c) // 2, (W - c) // 2),  # center
    ]
    crops = [im[y:y + c, x:x + c] for y, x in boxes]
    crops += [c_img[:, ::-1, :] for c_img in crops]  # horizontal flips
    return crops  # 10 crops


def compute_all_crops_and_features(records, ref_paths, device):
    inception = InceptionFeatureExtractor(device)
    clip_model = ClipEmbeddingModel(device)

    def feats_for(path):
        crops = five_crop_and_flip(path)
        fid_feats = inception.features_for_crops(crops)   # (10, 2048)
        cmmd_embs = clip_model.embed_crops(crops)          # (10, embed_dim)
        return fid_feats, cmmd_embs

    ref_fid, ref_cmmd = {}, {}
    n = len(ref_paths)
    for i, (key, path) in enumerate(ref_paths.items()):
        ref_fid[key], ref_cmmd[key] = feats_for(path)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  ref crops+features [{i+1}/{n}]")

    fake_fid, fake_cmmd = {}, {}
    n = len(records)
    for i, r in enumerate(records):
        key = (r["method"], r["rate"], r["scene"], r["idx"])
        fake_fid[key], fake_cmmd[key] = feats_for(r["path"])
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  fake crops+features [{i+1}/{n}]")

    return fake_fid, fake_cmmd, ref_fid, ref_cmmd


def compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd):
    by_scene_fid = defaultdict(list)
    by_scene_cmmd = defaultdict(list)
    for r in records:
        key = (r["method"], r["rate"], r["scene"], r["idx"])
        by_scene_fid[(r["method"], r["rate"], r["scene"])].append(fake_fid[key])
        by_scene_cmmd[(r["method"], r["rate"], r["scene"])].append(fake_cmmd[key])

    ref_fid_by_scene = defaultdict(list)
    ref_cmmd_by_scene = defaultdict(list)
    for (scene, idx), feat in ref_fid.items():
        ref_fid_by_scene[scene].append(feat)
    for (scene, idx), emb in ref_cmmd.items():
        ref_cmmd_by_scene[scene].append(emb)

    results = {}
    for (method, rate, scene), feat_list in by_scene_fid.items():
        fake_arr = np.concatenate(feat_list, axis=0)
        ref_arr = np.concatenate(ref_fid_by_scene[scene], axis=0)
        mu1, sig1 = fake_arr.mean(0), np.cov(fake_arr, rowvar=False)
        mu2, sig2 = ref_arr.mean(0), np.cov(ref_arr, rowvar=False)
        fid_val = frechet_distance(mu1, sig1, mu2, sig2)

        emb_list = by_scene_cmmd[(method, rate, scene)]
        fake_emb = np.concatenate(emb_list, axis=0)
        ref_emb = np.concatenate(ref_cmmd_by_scene[scene], axis=0)
        cmmd_val = mmd(fake_emb.astype(np.float32), ref_emb.astype(np.float32))

        results[(method, rate, scene)] = (fid_val, cmmd_val, fake_arr.shape[0], ref_arr.shape[0])
        print(f"  [{method}/{rate}/{scene}] FID={fid_val:.4f}  CMMD={cmmd_val:.4f}  "
              f"(n_fake_crops={fake_arr.shape[0]}, n_ref_crops={ref_arr.shape[0]})")

    return results


def write_outputs(per_scene, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / "fivecrop_per_scene.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rate", "scene", "fid", "cmmd", "n_fake_crops", "n_ref_crops"])
        w.writeheader()
        for (method, rate, scene), (fid_val, cmmd_val, n1, n2) in sorted(per_scene.items()):
            w.writerow({"method": method, "rate": rate, "scene": scene,
                        "fid": fid_val, "cmmd": cmmd_val, "n_fake_crops": n1, "n_ref_crops": n2})
    print(f"Wrote {p}")

    groups = defaultdict(list)
    for (method, rate, scene), (fid_val, cmmd_val, n1, n2) in per_scene.items():
        groups[(method, rate, "all9")].append((fid_val, cmmd_val))
        if scene in INDOOR:
            groups[(method, rate, "indoor")].append((fid_val, cmmd_val))
        elif scene in OUTDOOR:
            groups[(method, rate, "outdoor")].append((fid_val, cmmd_val))

    p = out_dir / "fivecrop_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "rate", "group", "fid_mean", "cmmd_mean", "n_scenes"])
        w.writeheader()
        for (method, rate, group), vals in sorted(groups.items()):
            fids = [v[0] for v in vals]
            cmmds = [v[1] for v in vals]
            w.writerow({"method": method, "rate": rate, "group": group,
                        "fid_mean": sum(fids) / len(fids), "cmmd_mean": sum(cmmds) / len(cmmds),
                        "n_scenes": len(vals)})
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
    print(f"Total (method,rate,view) records: {len(records)} (each -> 10 five-crop+flip sub-images)")

    print("Computing five-crop Inception (FID) + CLIP (CMMD) features for all fake + reference images...")
    fake_fid, fake_cmmd, ref_fid, ref_cmmd = compute_all_crops_and_features(records, ref_paths, args.device)

    print("Computing per-scene FID + CMMD...")
    per_scene = compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd)

    write_outputs(per_scene, args.out_dir)


if __name__ == "__main__":
    main()
