#!/usr/bin/env python
"""
Full-reference metrics (PSNR, SSIM, LPIPS, DISTS) + normal five-crop FID + model size
for methods in the tg_compare layout ({scene}_{idx}/gt.png + {method}.png).

Conventions (3DGS-style): PSNR on RGB; SSIM = pyiqa 'ssimc' (per-channel RGB, not the
Y-channel 'ssim'); LPIPS = VGG backbone ('lpips-vgg'); DISTS default. Every metric is
averaged over views within a scene, then over scenes (same aggregation as the FID).
FID is read from an existing normal (downsampled) per-scene CSV, not recomputed.
Size(MB) = 1e6 bytes, read from the dataset README ('size' column of the size-matched
table = textured model size); the RefinementNet adds 7,084 bytes on top.

Usage:
  python eval_fr_tgcompare.py --root <.../tg_compare_100k> --methods 2dgs_textured,2dgs_textured_crn_wd \
      --fid_csv results/tg_compare_100k/tgcompare_per_scene.csv --out_dir results/tg_compare_texture
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_fivecrop_tgcompare import discover  # noqa: E402

METRICS = {"psnr": "psnr", "ssim": "ssimc", "lpips": "lpips-vgg", "dists": "dists"}
CRN_BYTES = 7084


def read_sizes(readme):
    sizes = {}
    for line in Path(readme).read_text().splitlines():
        m = re.match(r"^(\w+)\s+([\d.]+) MB\s+[\d,]+\s+[\d,]+\s*$", line.strip())
        if m:
            sizes[m.group(1)] = float(m.group(2))
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--methods", default="2dgs_textured,2dgs_textured_crn_wd")
    ap.add_argument("--fid_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_per_scene", type=int, default=None)
    args = ap.parse_args()

    import torch
    import pyiqa

    root = Path(args.root)
    methods = args.methods.split(",")
    scenes, ref_paths = discover(root, None, args.limit_per_scene)
    models = {k: pyiqa.create_metric(v, device=args.device) for k, v in METRICS.items()}

    vals = defaultdict(lambda: defaultdict(list))  # (method, scene) -> metric -> [values]
    for method in methods:
        for scene, idxs in scenes.items():
            for idx in idxs:
                p = root / f"{scene}_{idx}" / f"{method}.png"
                ref = ref_paths[(scene, idx)]
                for k, m in models.items():
                    with torch.no_grad():
                        vals[(method, scene)][k].append(float(m(str(p), str(ref)).item()))
            print(f"  done {method}/{scene} ({len(idxs)} views)", flush=True)

    fid = {}
    with open(args.fid_csv) as f:
        for row in csv.DictReader(f):
            fid[(row["method"], row["scene"])] = float(row["fid"])

    sizes = read_sizes(root / "README.txt")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (method, scene), d in sorted(vals.items()):
        row = {"method": method, "scene": scene, "n_views": len(d["psnr"])}
        for k in METRICS:
            row[k] = sum(d[k]) / len(d[k])
        row["fid"] = fid.get((method, scene))
        size = sizes.get(scene)
        row["size_mb"] = None if size is None else size + (CRN_BYTES / 1e6 if method.endswith("crn_wd") else 0.0)
        rows.append(row)

    fields = ["method", "scene", "n_views", "lpips", "dists", "fid", "psnr", "ssim", "size_mb"]
    with open(out / "texture_per_scene.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    with open(out / "texture_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_scenes", "lpips", "dists", "fid", "psnr", "ssim",
                                          "size_mb_mean", "size_mb_total"])
        w.writeheader()
        for method, rs in by_method.items():
            n = len(rs)
            w.writerow({
                "method": method, "n_scenes": n,
                **{k: sum(r[k] for r in rs) / n for k in ("lpips", "dists", "fid", "psnr", "ssim")},
                "size_mb_mean": sum(r["size_mb"] for r in rs) / n,
                "size_mb_total": sum(r["size_mb"] for r in rs),
            })
    print(f"Wrote {out}/texture_per_scene.csv and texture_summary.csv")


if __name__ == "__main__":
    main()
