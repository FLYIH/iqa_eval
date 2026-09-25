#!/usr/bin/env python
"""
FID + CMMD using HiFiC-style patch extraction (eval_fid_patch.py's extract_patches:
two-grid, non-overlapping 256x256 patches, see its own docstring for the exact rule)
as the augmentation/sampling unit -- instead of the five-crop+flip scheme used by
eval_fid_cmmd_fivecrop.py -- for the "flat method, no rate axis" data layout (see
eval_fivecrop_tgcompare.py: {root}/{scene}_{idx}/{ref_name}, {method}.png).

Both FID (Inception-v3, eval_fid_patch.InceptionFeatureExtractor) and CMMD
(CLIP-ViT-L/14-336, eval_cmmd.ClipEmbeddingModel) are computed on the SAME patch
set per image, same as eval_fid_cmmd_fivecrop.py does for five-crop.

--no_downsample skips resizing each 256x256 patch to the models' native input
(299x299 for Inception, 336x336 for CLIP) and feeds it at native 256x256 instead
(Inception via adaptive avg pool; CLIP via interpolate_pos_encoding=True -- note
256 is not an exact multiple of CLIP's 14px patch size, so the patch grid is
floor(256/14)=18x18=324 patches, with a few edge pixels dropped by the conv, same
as any other non-multiple-of-14 input).

Usage:
  python eval_fid_cmmd_patch_tgcompare.py --root /work/u4546465/evaluation/Perceptual_gs \
      --ref_name reference.png --out_dir /home/u4546465/iqa_eval/results/Perceptual_gs \
      --no_downsample --device cuda
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_fivecrop_tgcompare import discover, build_records  # noqa: E402
from eval_fid_cmmd_fivecrop import INDOOR, OUTDOOR  # noqa: E402
from eval_fid_patch import InceptionFeatureExtractor, frechet_distance, extract_patches  # noqa: E402
from eval_cmmd import ClipEmbeddingModel, mmd  # noqa: E402


def compute_all_patches_and_features(records, ref_paths, device, no_downsample):
    inception = InceptionFeatureExtractor(device)
    clip_model = ClipEmbeddingModel(device)

    def feats_for(path):
        patches = extract_patches(str(path))
        if no_downsample:
            return inception.features_for_crops_raw(patches), clip_model.embed_crops_raw(patches)
        return inception.features_for_crops(patches), clip_model.embed_crops(patches)

    ref_fid, ref_cmmd = {}, {}
    n = len(ref_paths)
    for i, (key, path) in enumerate(ref_paths.items()):
        ref_fid[key], ref_cmmd[key] = feats_for(path)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  ref patches+features [{i+1}/{n}]", flush=True)

    fake_fid, fake_cmmd = {}, {}
    n = len(records)
    for i, r in enumerate(records):
        key = (r["method"], r["scene"], r["idx"])
        try:
            fake_fid[key], fake_cmmd[key] = feats_for(r["path"])
        except OSError as e:
            print(f"WARNING: skipping unreadable image {r['path']}: {e}", file=sys.stderr)
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  fake patches+features [{i+1}/{n}]", flush=True)

    return fake_fid, fake_cmmd, ref_fid, ref_cmmd


def compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd):
    by_scene_fid = defaultdict(list)
    by_scene_cmmd = defaultdict(list)
    for r in records:
        key = (r["method"], r["scene"], r["idx"])
        if key not in fake_fid:
            continue
        by_scene_fid[(r["method"], r["scene"])].append(fake_fid[key])
        by_scene_cmmd[(r["method"], r["scene"])].append(fake_cmmd[key])

    ref_fid_by_scene = defaultdict(list)
    ref_cmmd_by_scene = defaultdict(list)
    for (scene, idx), feat in ref_fid.items():
        ref_fid_by_scene[scene].append(feat)
    for (scene, idx), emb in ref_cmmd.items():
        ref_cmmd_by_scene[scene].append(emb)

    results = {}
    for (method, scene), feat_list in by_scene_fid.items():
        fake_arr = np.concatenate(feat_list, axis=0)
        ref_arr = np.concatenate(ref_fid_by_scene[scene], axis=0)
        mu1, sig1 = fake_arr.mean(0), np.cov(fake_arr, rowvar=False)
        mu2, sig2 = ref_arr.mean(0), np.cov(ref_arr, rowvar=False)
        fid_val = frechet_distance(mu1, sig1, mu2, sig2)

        emb_list = by_scene_cmmd[(method, scene)]
        fake_emb = np.concatenate(emb_list, axis=0)
        ref_emb = np.concatenate(ref_cmmd_by_scene[scene], axis=0)
        cmmd_val = mmd(fake_emb.astype(np.float32), ref_emb.astype(np.float32))

        results[(method, scene)] = (fid_val, cmmd_val, fake_arr.shape[0], ref_arr.shape[0])
        print(f"  [{method}/{scene}] FID={fid_val:.4f}  CMMD={cmmd_val:.4f}  "
              f"(n_fake_patches={fake_arr.shape[0]}, n_ref_patches={ref_arr.shape[0]})", flush=True)

    return results


def write_outputs(per_scene, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / f"{prefix}_per_scene.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scene", "fid", "cmmd", "n_fake_patches", "n_ref_patches"])
        w.writeheader()
        for (method, scene), (fid_val, cmmd_val, n1, n2) in sorted(per_scene.items()):
            w.writerow({"method": method, "scene": scene, "fid": fid_val, "cmmd": cmmd_val,
                        "n_fake_patches": n1, "n_ref_patches": n2})
    print(f"Wrote {p}")

    groups = defaultdict(list)
    for (method, scene), (fid_val, cmmd_val, n1, n2) in per_scene.items():
        groups[(method, "all9")].append((fid_val, cmmd_val))
        if scene in INDOOR:
            groups[(method, "indoor")].append((fid_val, cmmd_val))
        elif scene in OUTDOOR:
            groups[(method, "outdoor")].append((fid_val, cmmd_val))

    p = out_dir / f"{prefix}_summary.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "group", "fid_mean", "cmmd_mean", "n_scenes"])
        w.writeheader()
        for (method, group), vals in sorted(groups.items()):
            fids = [v[0] for v in vals]
            cmmds = [v[1] for v in vals]
            w.writerow({"method": method, "group": group,
                        "fid_mean": sum(fids) / len(fids), "cmmd_mean": sum(cmmds) / len(cmmds),
                        "n_scenes": len(vals)})
    print(f"Wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref_name", default="gt.png")
    ap.add_argument("--out_dir", default="/home/u4546465/iqa_eval/results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_scenes", default=None)
    ap.add_argument("--limit_per_scene", type=int, default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--no_downsample", action="store_true")
    args = ap.parse_args()

    limit_scenes = set(args.limit_scenes.split(",")) if args.limit_scenes else None
    methods = set(args.methods.split(",")) if args.methods else None

    print("Discovering views...")
    scenes, ref_paths = discover(args.root, limit_scenes, args.limit_per_scene, ref_name=args.ref_name)
    print(f"Found {len(scenes)} scenes, {sum(len(v) for v in scenes.values())} views total")

    print(f"Building file records (methods={methods or 'ALL'})...")
    records = build_records(args.root, scenes, ref_name=args.ref_name, methods=methods)
    print(f"Total (method,view) records: {len(records)} (each -> HiFiC two-grid 256x256 patches)")

    print(f"Computing HiFiC-patch Inception (FID) + CLIP (CMMD) features "
          f"(no_downsample={args.no_downsample})...")
    fake_fid, fake_cmmd, ref_fid, ref_cmmd = compute_all_patches_and_features(
        records, ref_paths, args.device, args.no_downsample)

    print("Computing per-scene FID + CMMD...")
    per_scene = compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd)

    prefix = "patch_nodownsample" if args.no_downsample else "patch"
    write_outputs(per_scene, args.out_dir, prefix)


if __name__ == "__main__":
    main()
