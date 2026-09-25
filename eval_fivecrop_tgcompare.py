#!/usr/bin/env python
"""
Five-crop FID+CMMD (same protocol/code as eval_fid_cmmd_fivecrop.py) for "flat
method, no rate axis" data layouts that differ from CAT3DGSPro_organized's
{method}@{rate}.png convention:

  {root}/{scene}_{idx}/
      {ref_name}          <- ground truth, e.g. "gt.png" or "reference.png"
      {method1}.png        <- one file per method, no @rate suffix
      {method2}.png
      ...

Methods are auto-discovered per scene directory (every *.png that isn't
ref_name); pass --methods to restrict to a subset. --ref_name defaults to
"gt.png" (tg_compare_100k*); pass "reference.png" for Perceptual_gs etc.

Usage:
  python eval_fivecrop_tgcompare.py --root /work/u4546465/evaluation/tg_compare_100k/tg_compare_100k \
      --out_dir /home/u4546465/iqa_eval/results/tg_compare_100k --no_downsample --device cuda

  python eval_fivecrop_tgcompare.py --root /work/u4546465/evaluation/Perceptual_gs \
      --ref_name reference.png \
      --out_dir /home/u4546465/iqa_eval/results/Perceptual_gs --no_downsample --device cuda
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_fid_cmmd_fivecrop import five_crop_and_flip, INDOOR, OUTDOOR  # noqa: E402
from eval_fid_patch import InceptionFeatureExtractor, frechet_distance  # noqa: E402
from eval_cmmd import ClipEmbeddingModel, mmd  # noqa: E402

SCENE_RE = re.compile(r"^(?P<scene>.+)_(?P<idx>\d+)$")


def discover(root, limit_scenes=None, limit_per_scene=None, ref_name="gt.png"):
    root = Path(root)
    scenes = defaultdict(list)
    ref_paths = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = SCENE_RE.match(d.name)
        if not m:
            continue
        scene, idx = m.group("scene"), m.group("idx")
        if limit_scenes and scene not in limit_scenes:
            continue
        ref = d / ref_name
        if not ref.exists():
            print(f"WARNING: missing {ref_name} in {d}", file=sys.stderr)
            continue
        scenes[scene].append(idx)
        ref_paths[(scene, idx)] = ref

    for scene in scenes:
        scenes[scene].sort()
        if limit_per_scene:
            scenes[scene] = scenes[scene][:limit_per_scene]

    valid_keys = {(scene, idx) for scene, idxs in scenes.items() for idx in idxs}
    ref_paths = {key: path for key, path in ref_paths.items() if key in valid_keys}
    return scenes, ref_paths


def build_records(root, scenes, ref_name="gt.png", methods=None):
    root = Path(root)
    _, ref_paths = discover(root, ref_name=ref_name)
    records = []
    for scene, idxs in scenes.items():
        for idx in idxs:
            ref_path = ref_paths[(scene, idx)]
            scene_dir = root / f"{scene}_{idx}"
            found_methods = sorted(p.stem for p in scene_dir.glob("*.png") if p.name != ref_name)
            use_methods = [m for m in found_methods if methods is None or m in methods]
            for method in use_methods:
                p = scene_dir / f"{method}.png"
                if p.exists():
                    records.append(dict(method=method, scene=scene, idx=idx, path=p, ref_path=ref_path))
                else:
                    print(f"WARNING: missing {p}", file=sys.stderr)
    return records


def compute_all_crops_and_features(records, ref_paths, device, no_downsample):
    inception = InceptionFeatureExtractor(device)
    clip_model = ClipEmbeddingModel(device)

    def feats_for(path):
        crops = five_crop_and_flip(path)
        if no_downsample:
            return inception.features_for_crops_raw(crops), clip_model.embed_crops_raw(crops)
        return inception.features_for_crops(crops), clip_model.embed_crops(crops)

    ref_fid, ref_cmmd = {}, {}
    n = len(ref_paths)
    for i, (key, path) in enumerate(ref_paths.items()):
        ref_fid[key], ref_cmmd[key] = feats_for(path)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  ref crops+features [{i+1}/{n}]")

    fake_fid, fake_cmmd = {}, {}
    n = len(records)
    for i, r in enumerate(records):
        key = (r["method"], r["scene"], r["idx"])
        fake_fid[key], fake_cmmd[key] = feats_for(r["path"])
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  fake crops+features [{i+1}/{n}]")

    return fake_fid, fake_cmmd, ref_fid, ref_cmmd


def compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd):
    by_scene_fid = defaultdict(list)
    by_scene_cmmd = defaultdict(list)
    for r in records:
        key = (r["method"], r["scene"], r["idx"])
        by_scene_fid[(r["method"], r["scene"])].append(fake_fid[key])
        by_scene_cmmd[(r["method"], r["scene"])].append(fake_cmmd[key])

    ref_fid_by_scene = defaultdict(list)
    ref_cmmd_by_scene = defaultdict(list)
    for (scene, idx), feat in ref_fid.items():
        ref_fid_by_scene[scene].append(feat)
    for (scene, idx), emb in ref_cmmd.items():
        ref_cmmd_by_scene[scene].append(emb)

    import numpy as np
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
              f"(n_fake_crops={fake_arr.shape[0]}, n_ref_crops={ref_arr.shape[0]})")

    return results


def write_outputs(per_scene, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / f"{prefix}_per_scene.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "scene", "fid", "cmmd", "n_fake_crops", "n_ref_crops"])
        w.writeheader()
        for (method, scene), (fid_val, cmmd_val, n1, n2) in sorted(per_scene.items()):
            w.writerow({"method": method, "scene": scene, "fid": fid_val, "cmmd": cmmd_val,
                        "n_fake_crops": n1, "n_ref_crops": n2})
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
    ap.add_argument("--out_dir", default="/home/u4546465/iqa_eval/results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_scenes", default=None)
    ap.add_argument("--limit_per_scene", type=int, default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--ref_name", default="gt.png",
                    help="Ground-truth filename inside each scene dir (default 'gt.png'; "
                         "use 'reference.png' for e.g. Perceptual_gs).")
    ap.add_argument("--no_downsample", action="store_true")
    args = ap.parse_args()

    limit_scenes = set(args.limit_scenes.split(",")) if args.limit_scenes else None
    methods = set(args.methods.split(",")) if args.methods else None

    print("Discovering views...")
    scenes, ref_paths = discover(args.root, limit_scenes, args.limit_per_scene, ref_name=args.ref_name)
    print(f"Found {len(scenes)} scenes, {sum(len(v) for v in scenes.values())} views total")

    print(f"Building file records (methods={methods or 'ALL'})...")
    records = build_records(args.root, scenes, ref_name=args.ref_name, methods=methods)
    print(f"Total (method,view) records: {len(records)} (each -> 10 five-crop+flip sub-images)")

    print(f"Computing five-crop Inception (FID) + CLIP (CMMD) features "
          f"(no_downsample={args.no_downsample})...")
    fake_fid, fake_cmmd, ref_fid, ref_cmmd = compute_all_crops_and_features(
        records, ref_paths, args.device, args.no_downsample)

    print("Computing per-scene FID + CMMD...")
    per_scene = compute_per_scene(records, ref_fid, ref_cmmd, fake_fid, fake_cmmd)

    prefix = "tgcompare_nodownsample" if args.no_downsample else "tgcompare"
    write_outputs(per_scene, args.out_dir, prefix)


if __name__ == "__main__":
    main()
